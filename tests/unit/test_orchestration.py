"""Unit tests for detector orchestration: ``run_detectors`` and ``AgentLens.detect``."""

from collections.abc import Sequence
from uuid import uuid4

import pytest

from agentlens import AgentLens, AgentLensError
from agentlens.detectors import (
    DEFAULT_DETECTOR_CLASSES,
    DuplicateToolDetector,
    InefficiencyDetector,
    IssueDetector,
    LoopDetector,
    RetryDetector,
    run_detectors,
)
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType

_RUN = AgentRun(task="synthetic orchestration trace")


def _ev(seq, event_type, name, **kw):
    return AgentEvent(run_id=_RUN.id, sequence_number=seq, event_type=event_type, name=name, **kw)


def _content(issue: AgentIssue) -> dict:
    """Issue content minus the freshly generated id."""

    return issue.model_dump(mode="json", exclude={"id"})


def _loop_then_duplicate_events() -> list[AgentEvent]:
    """A trace with a 3x decision loop followed by an unnecessary duplicate call."""

    events = []
    seq = 0
    # loop: analyze / search x3
    for _ in range(3):
        events.append(_ev(seq, EventType.DECISION, "analyze_request", input={"q": "x"}))
        events.append(_ev(seq + 1, EventType.DECISION, "search_for_information", input={"q": "x"}))
        seq += 2
    # duplicate successful tool call (same name + input, no new info between)
    for _ in range(2):
        events.append(_ev(seq, EventType.TOOL_CALL_STARTED, "lookup", input={"id": "1"}))
        events.append(
            _ev(
                seq + 1,
                EventType.TOOL_CALL_COMPLETED,
                "lookup",
                input={"id": "1"},
                output={"ok": 1},
            )
        )
        seq += 2
    return events


# ---------------------------------------------------------------------------
# A. Public API
# ---------------------------------------------------------------------------


def test_run_detectors_is_importable_and_returns_list():
    result = run_detectors(_RUN, [])
    assert isinstance(result, list)
    assert result == []


def test_default_detector_classes_are_the_four_in_fixed_order():
    assert (
        LoopDetector,
        RetryDetector,
        DuplicateToolDetector,
        InefficiencyDetector,
    ) == DEFAULT_DETECTOR_CLASSES
    for cls in DEFAULT_DETECTOR_CLASSES:
        assert isinstance(cls(), IssueDetector)


def test_agentlens_has_detect_returning_list_of_issues():
    lens = AgentLens()
    with lens.trace("noop"):
        pass
    run_id = lens.list_runs()[0].id
    issues = lens.detect(run_id)
    assert isinstance(issues, list)
    assert all(isinstance(i, AgentIssue) for i in issues)


def test_detect_accepts_a_run_object_too():
    lens = AgentLens()
    with lens.trace("noop"):
        pass
    run = lens.list_runs()[0]
    a = [_content(i) for i in lens.detect(run)]
    b = [_content(i) for i in lens.detect(run.id)]
    assert a == b


def test_detect_unknown_run_raises_agentlens_error():
    lens = AgentLens()
    with pytest.raises(AgentLensError, match="no run with id"):
        lens.detect(uuid4())


def test_run_detectors_matches_the_issue_detector_protocol_shape():
    # run_detectors is a function, but each class it drives satisfies the protocol
    class Wrapper:
        def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
            return run_detectors(run, events)

    assert isinstance(Wrapper(), IssueDetector)


# ---------------------------------------------------------------------------
# C. Combined-detector equivalence
# ---------------------------------------------------------------------------


def test_run_detectors_equals_manual_concatenation():
    events = _loop_then_duplicate_events()
    manual = [
        *LoopDetector().detect(_RUN, events),
        *RetryDetector().detect(_RUN, events),
        *DuplicateToolDetector().detect(_RUN, events),
        *InefficiencyDetector().detect(_RUN, events),
    ]
    combined = run_detectors(_RUN, events)
    assert [_content(i) for i in combined] == [_content(i) for i in manual]


def test_detect_equals_manual_concatenation_for_stored_run():
    lens = AgentLens()
    with lens.trace("loop then dup") as trace:
        for _ in range(3):
            trace.record_event(
                event_type=EventType.DECISION, name="analyze_request", input={"q": 1}
            )
            trace.record_event(
                event_type=EventType.DECISION, name="search_for_information", input={"q": 1}
            )
        for _ in range(2):
            trace.record_event(event_type=EventType.TOOL_CALL_STARTED, name="lk", input={"i": 1})
            trace.record_event(
                event_type=EventType.TOOL_CALL_COMPLETED, name="lk", input={"i": 1}, output={"o": 1}
            )
    run = lens.get_run(trace.run_id)
    events = lens.get_events(trace.run_id)

    expected = [
        *LoopDetector().detect(run, events),
        *RetryDetector().detect(run, events),
        *DuplicateToolDetector().detect(run, events),
        *InefficiencyDetector().detect(run, events),
    ]
    assert [_content(i) for i in lens.detect(trace.run_id)] == [_content(i) for i in expected]


# ---------------------------------------------------------------------------
# D. Ordering
# ---------------------------------------------------------------------------


def test_issue_order_follows_fixed_detector_order():
    events = _loop_then_duplicate_events()
    issues = run_detectors(_RUN, events)
    types = [i.issue_type for i in issues]
    assert types == [IssueType.AGENT_LOOP, IssueType.DUPLICATE_TOOL_CALL]


def test_multiple_issues_from_one_detector_keep_trace_order():
    events = []
    seq = 0
    # two separate duplicate regions -> two DuplicateTool issues in trace order
    for tool in ("alpha", "beta"):
        for _ in range(2):
            events.append(_ev(seq, EventType.TOOL_CALL_STARTED, tool, input={"x": 1}))
            events.append(
                _ev(seq + 1, EventType.TOOL_CALL_COMPLETED, tool, input={"x": 1}, output={"o": 1})
            )
            seq += 2
        events.append(_ev(seq, EventType.LLM_CALL_COMPLETED, "boundary"))
        seq += 1

    issues = run_detectors(_RUN, events)
    assert [i.issue_type for i in issues] == [
        IssueType.DUPLICATE_TOOL_CALL,
        IssueType.DUPLICATE_TOOL_CALL,
    ]
    assert [i.metadata["operation"] for i in issues] == ["alpha", "beta"]


def test_detection_is_deterministic_across_calls_and_shuffled_input():
    events = _loop_then_duplicate_events()
    first = [_content(i) for i in run_detectors(_RUN, events)]
    second = [_content(i) for i in run_detectors(_RUN, list(reversed(events)))]
    assert first == second


# ---------------------------------------------------------------------------
# E. Cross-detector interaction
# ---------------------------------------------------------------------------


def _retry_then_duplicate_events() -> list[AgentEvent]:
    """Retry streak (3 failed attempts) followed by an unrelated duplicate call.

    The two regions do not overlap, so the two detectors' findings must be
    independent -- distinct issue types, in detector order, with disjoint
    related-event sets.
    """

    events = []
    seq = 0
    for _ in range(3):
        events.append(_ev(seq, EventType.TOOL_CALL_STARTED, "fetch", input={"id": "1"}))
        events.append(_ev(seq + 1, EventType.ERROR, "temp_fail", output={"operation": "fetch"}))
        seq += 2
    for _ in range(2):
        events.append(_ev(seq, EventType.TOOL_CALL_STARTED, "lookup", input={"id": "9"}))
        events.append(
            _ev(
                seq + 1,
                EventType.TOOL_CALL_COMPLETED,
                "lookup",
                input={"id": "9"},
                output={"ok": 1},
            )
        )
        seq += 2
    return events


def test_multi_detector_trace_returns_independent_findings_in_detector_order():
    events = _retry_then_duplicate_events()
    issues = run_detectors(_RUN, events)

    assert [i.issue_type for i in issues] == [
        IssueType.EXCESSIVE_RETRY,
        IssueType.DUPLICATE_TOOL_CALL,
    ]
    # equivalent to manual concatenation
    manual = [
        *LoopDetector().detect(_RUN, events),
        *RetryDetector().detect(_RUN, events),
        *DuplicateToolDetector().detect(_RUN, events),
        *InefficiencyDetector().detect(_RUN, events),
    ]
    assert [_content(i) for i in issues] == [_content(i) for i in manual]
    # the two findings touch disjoint events (non-overlapping regions)
    retry_ids = set(issues[0].related_event_ids)
    dup_ids = set(issues[1].related_event_ids)
    assert retry_ids.isdisjoint(dup_ids)
    assert all(eid in {e.id for e in events} for eid in retry_ids | dup_ids)


def test_every_orchestrated_issue_round_trips_through_json():
    events = _loop_then_duplicate_events()
    for issue in run_detectors(_RUN, events):
        restored = AgentIssue.model_validate_json(issue.model_dump_json())
        assert restored == issue


# ---------------------------------------------------------------------------
# F. Purity
# ---------------------------------------------------------------------------


def test_run_detectors_does_not_mutate_inputs():
    events = _loop_then_duplicate_events()
    run = AgentRun(task="keep intact")
    events_before = [e.model_dump() for e in events]
    run_before = run.model_dump()
    order_before = [e.sequence_number for e in events]

    run_detectors(run, events)

    assert [e.model_dump() for e in events] == events_before
    assert run.model_dump() == run_before
    assert [e.sequence_number for e in events] == order_before


def test_detect_does_not_change_run_status_or_stored_events():
    lens = AgentLens()
    with lens.trace("t") as trace:
        trace.record_event(event_type=EventType.DECISION, name="d", input={})
    run_id = trace.run_id
    status_before = lens.get_run(run_id).status
    events_before = [e.model_dump() for e in lens.get_events(run_id)]

    lens.detect(run_id)
    lens.detect(run_id)  # repeated calls

    assert lens.get_run(run_id).status is status_before
    assert [e.model_dump() for e in lens.get_events(run_id)] == events_before

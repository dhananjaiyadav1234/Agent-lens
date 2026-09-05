"""Synthetic unit tests for :class:`agentlens.detectors.LoopDetector`.

These build ``AgentEvent`` / ``AgentRun`` models directly (not via the tracing
engine) so arbitrary event sequences can be exercised. The detector under test
still only ever sees universal models.
"""

import json
from copy import deepcopy
from uuid import UUID

import pytest

from agentlens.detectors import LoopDetector
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

_RUN = AgentRun(task="synthetic trace")


def _ev(seq, name, *, event_type=EventType.DECISION, input=None, metadata=None):
    return AgentEvent(
        run_id=_RUN.id,
        sequence_number=seq,
        event_type=event_type,
        name=name,
        input=input,
        metadata=metadata or {},
    )


def _sequence(names, *, event_type=EventType.DECISION, start=0):
    return [_ev(start + i, name, event_type=event_type) for i, name in enumerate(names)]


def _detect(events, **kwargs):
    return LoopDetector(**kwargs).detect(_RUN, events)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def test_single_event_loop():
    issues = _detect(_sequence(["a", "a", "a"]))
    assert len(issues) == 1
    assert issues[0].metadata["cycle_length"] == 1
    assert issues[0].metadata["repetitions"] == 3
    assert len(issues[0].related_event_ids) == 3


def test_two_event_loop():
    issues = _detect(_sequence(["a", "b", "a", "b", "a", "b"]))
    assert len(issues) == 1
    assert issues[0].metadata["cycle_length"] == 2
    assert issues[0].metadata["repetitions"] == 3
    assert len(issues[0].related_event_ids) == 6
    assert issues[0].metadata["pattern"] == [
        {"event_type": "decision", "name": "a"},
        {"event_type": "decision", "name": "b"},
    ]


def test_incomplete_cycle_is_not_a_loop():
    assert _detect(_sequence(["a", "b", "a", "b", "a"])) == []


def test_below_threshold_is_not_a_loop():
    assert _detect(_sequence(["a", "b", "a", "b"])) == []


def test_min_repetitions_two_detects_double_cycle():
    issues = _detect(_sequence(["a", "b", "a", "b"]), minimum_repetitions=2)
    assert len(issues) == 1
    assert issues[0].metadata["cycle_length"] == 2
    assert issues[0].metadata["repetitions"] == 2


def test_trailing_events_after_a_loop_are_excluded():
    issues = _detect(_sequence(["a", "b", "a", "b", "a", "b", "a"]))
    assert len(issues) == 1
    assert len(issues[0].related_event_ids) == 6


def test_leading_events_before_a_loop_are_excluded():
    issues = _detect(_sequence(["start", "a", "a", "a"]))
    assert len(issues) == 1
    assert len(issues[0].related_event_ids) == 3


# ---------------------------------------------------------------------------
# Overlapping / nested policy
# ---------------------------------------------------------------------------


def test_run_of_six_identical_events_is_one_issue_shortest_cycle():
    issues = _detect(_sequence(["a"] * 6))
    assert len(issues) == 1
    assert issues[0].metadata["cycle_length"] == 1
    assert issues[0].metadata["repetitions"] == 6
    assert len(issues[0].related_event_ids) == 6


def test_ababab_prefers_len_two_cycle_over_treating_a_and_b_separately():
    issues = _detect(_sequence(["a", "b"] * 3))
    assert len(issues) == 1
    assert issues[0].metadata["cycle_length"] == 2


# ---------------------------------------------------------------------------
# Fingerprint-driven behaviour
# ---------------------------------------------------------------------------


def test_different_inputs_are_not_a_loop():
    events = [
        _ev(0, "a", input={"n": 1}),
        _ev(1, "a", input={"n": 2}),
        _ev(2, "a", input={"n": 3}),
    ]
    assert _detect(events) == []


def test_equivalent_dict_ordering_is_a_loop():
    events = [
        _ev(0, "a", input={"x": 1, "y": 2}),
        _ev(1, "a", input={"y": 2, "x": 1}),
        _ev(2, "a", input={"x": 1, "y": 2}),
    ]
    issues = _detect(events)
    assert len(issues) == 1
    assert issues[0].metadata["repetitions"] == 3


def test_metadata_differences_do_not_break_a_loop():
    events = [
        _ev(0, "analyze", input={"q": "x"}, metadata={"iteration": 0}),
        _ev(1, "analyze", input={"q": "x"}, metadata={"iteration": 1}),
        _ev(2, "analyze", input={"q": "x"}, metadata={"iteration": 2}),
    ]
    issues = _detect(events)
    assert len(issues) == 1
    assert issues[0].metadata["repetitions"] == 3


# ---------------------------------------------------------------------------
# Multiple independent loops
# ---------------------------------------------------------------------------


def test_two_separate_loops_produce_two_issues_in_trace_order():
    events = _sequence(["a", "b", "a", "b", "a", "b", "c", "d", "c", "d", "c", "d"])
    issues = _detect(events)
    assert len(issues) == 2

    first, second = issues
    assert first.metadata["pattern"] == [
        {"event_type": "decision", "name": "a"},
        {"event_type": "decision", "name": "b"},
    ]
    assert second.metadata["pattern"] == [
        {"event_type": "decision", "name": "c"},
        {"event_type": "decision", "name": "d"},
    ]

    first_ids = set(first.related_event_ids)
    second_ids = set(second.related_event_ids)
    assert first_ids.isdisjoint(second_ids)
    assert first.related_event_ids == [e.id for e in events[:6]]
    assert second.related_event_ids == [e.id for e in events[6:12]]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_unsorted_input_is_detected_and_caller_list_is_untouched():
    events = _sequence(["a", "b", "a", "b", "a", "b"])
    shuffled = [events[3], events[0], events[5], events[1], events[4], events[2]]
    original_order = list(shuffled)

    issues = LoopDetector().detect(_RUN, shuffled)

    assert len(issues) == 1
    assert issues[0].related_event_ids == [e.id for e in events]  # sorted order
    assert shuffled == original_order  # caller's list not reordered
    assert shuffled is not original_order


def test_ordering_uses_sequence_number_not_list_position():
    events = _sequence(["a", "b", "a", "b", "a", "b"])
    assert LoopDetector().detect(_RUN, list(reversed(events)))[0].metadata["cycle_length"] == 2


# ---------------------------------------------------------------------------
# Lifecycle handling
# ---------------------------------------------------------------------------


def test_lifecycle_events_are_excluded_from_the_loop():
    events = [
        _ev(0, "run_started", event_type=EventType.RUN_STARTED),
        *_sequence(["a", "b", "a", "b", "a", "b"], start=1),
        _ev(7, "run_completed", event_type=EventType.RUN_COMPLETED),
    ]
    issues = _detect(events)
    assert len(issues) == 1

    loop_event_ids = set(issues[0].related_event_ids)
    assert len(loop_event_ids) == 6
    assert events[0].id not in loop_event_ids
    assert events[-1].id not in loop_event_ids
    assert loop_event_ids == {e.id for e in events[1:7]}


def test_repeated_lifecycle_events_do_not_form_a_loop():
    events = [
        _ev(0, "run_started", event_type=EventType.RUN_STARTED),
        _ev(1, "run_started", event_type=EventType.RUN_STARTED),
        _ev(2, "run_started", event_type=EventType.RUN_STARTED),
    ]
    assert _detect(events) == []


# ---------------------------------------------------------------------------
# Retry-shaped sequences
# ---------------------------------------------------------------------------


def test_retry_shaped_sequence_is_not_a_loop():
    events = []
    seq = 0
    for _ in range(3):
        events.append(_ev(seq, "fetch", event_type=EventType.TOOL_CALL_STARTED, input={"id": "1"}))
        events.append(_ev(seq + 1, "temporary_failure", event_type=EventType.ERROR))
        seq += 2
    assert _detect(events) == []


def test_retry_then_success_sequence_is_not_a_loop():
    events = [
        _ev(0, "fetch", event_type=EventType.TOOL_CALL_STARTED, input={"id": "1"}),
        _ev(1, "err", event_type=EventType.ERROR),
        _ev(2, "fetch", event_type=EventType.TOOL_CALL_STARTED, input={"id": "1"}),
        _ev(3, "err", event_type=EventType.ERROR),
        _ev(4, "fetch", event_type=EventType.TOOL_CALL_STARTED, input={"id": "1"}),
        _ev(5, "err", event_type=EventType.ERROR),
        _ev(6, "fetch", event_type=EventType.TOOL_CALL_STARTED, input={"id": "1"}),
        _ev(7, "fetch", event_type=EventType.TOOL_CALL_COMPLETED, input={"id": "1"}),
    ]
    assert _detect(events) == []


def test_error_splits_an_otherwise_repeating_sequence():
    # a b a b <ERROR> a b -> no segment has 3 consecutive [a, b] cycles
    events = [
        _ev(0, "a"),
        _ev(1, "b"),
        _ev(2, "a"),
        _ev(3, "b"),
        _ev(4, "boom", event_type=EventType.ERROR),
        _ev(5, "a"),
        _ev(6, "b"),
    ]
    assert _detect(events) == []


# ---------------------------------------------------------------------------
# Severity policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repetitions", "expected"),
    [
        (3, Severity.MEDIUM),
        (4, Severity.HIGH),
        (5, Severity.HIGH),
        (6, Severity.CRITICAL),
        (9, Severity.CRITICAL),
    ],
)
def test_severity_scales_with_repetition_count(repetitions, expected):
    issues = _detect(_sequence(["a"] * repetitions))
    assert len(issues) == 1
    assert issues[0].severity is expected
    assert issues[0].metadata["repetitions"] == repetitions


# ---------------------------------------------------------------------------
# Empty / short traces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("events", [[], _sequence(["a"]), _sequence(["a", "b"])])
def test_empty_and_short_traces_return_no_issues(events):
    assert _detect(events) == []


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_default_minimum_repetitions_is_three():
    assert LoopDetector().minimum_repetitions == 3


@pytest.mark.parametrize("value", [1, 0, -1, -5])
def test_minimum_repetitions_below_two_is_rejected(value):
    with pytest.raises(ValueError, match="at least 2"):
        LoopDetector(minimum_repetitions=value)


@pytest.mark.parametrize("value", ["3", 3.0, None, True, 2.9])
def test_non_integer_minimum_repetitions_is_rejected(value):
    with pytest.raises(TypeError, match="must be an int"):
        LoopDetector(minimum_repetitions=value)


def test_string_three_is_not_coerced():
    with pytest.raises(TypeError):
        LoopDetector(minimum_repetitions="3")


# ---------------------------------------------------------------------------
# Input purity
# ---------------------------------------------------------------------------


def test_detection_does_not_mutate_events_or_nested_structures():
    nested_input = {"outer": {"inner": [1, 2, 3]}}
    events = [
        _ev(0, "a", input=deepcopy(nested_input)),
        _ev(1, "a", input=deepcopy(nested_input)),
        _ev(2, "a", input=deepcopy(nested_input)),
    ]
    before = [e.model_dump() for e in events]

    LoopDetector().detect(_RUN, events)

    after = [e.model_dump() for e in events]
    assert before == after
    assert events[0].input == nested_input


def test_detection_does_not_mutate_the_run():
    run = AgentRun(task="keep me intact")
    before = run.model_dump()
    LoopDetector().detect(run, _sequence(["a", "a", "a"]))
    assert run.model_dump() == before


# ---------------------------------------------------------------------------
# AgentIssue validity
# ---------------------------------------------------------------------------


def test_issues_are_valid_agent_issues():
    issues = _detect(_sequence(["a", "b"] * 3))
    assert issues
    for issue in issues:
        assert isinstance(issue, AgentIssue)
        assert issue.run_id == _RUN.id
        assert issue.issue_type is IssueType.LOOP
        assert issue.issue_type is IssueType.AGENT_LOOP
        assert issue.issue_type.value == "agent_loop"
        assert isinstance(issue.severity, Severity)
        assert all(isinstance(eid, UUID) for eid in issue.related_event_ids)
        assert issue.description


def test_issue_metadata_is_json_compatible():
    issue = _detect(_sequence(["a", "b"] * 3))[0]
    dumped = json.dumps(issue.metadata)
    assert json.loads(dumped) == issue.metadata


def test_issue_round_trips_through_json():
    issue = _detect(_sequence(["a", "b"] * 3))[0]
    restored = AgentIssue.model_validate_json(issue.model_dump_json())
    assert restored == issue


def test_related_event_ids_preserve_trace_order():
    events = _sequence(["a", "b", "a", "b", "a", "b"])
    issue = LoopDetector().detect(_RUN, events)[0]
    assert issue.related_event_ids == [e.id for e in events]


def test_detection_is_deterministic_in_content():
    events = _sequence(["a", "b", "c"] * 4)
    first = _detect(events)
    second = _detect(list(reversed(events)))
    assert len(first) == len(second) == 1
    assert first[0].metadata == second[0].metadata
    assert first[0].related_event_ids == second[0].related_event_ids
    assert first[0].description == second[0].description
    assert first[0].severity == second[0].severity

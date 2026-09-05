"""Synthetic unit tests for :class:`agentlens.detectors.DuplicateToolDetector`."""

import json
from collections.abc import Sequence
from copy import deepcopy
from uuid import UUID

import pytest

from agentlens.detectors import DuplicateToolDetector, IssueDetector
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

_RUN = AgentRun(task="synthetic trace")

_OP_A = ("lookup_customer", {"customer_id": "123"})
_OP_B = ("fetch_customer_context", {"customer_id": "123"})


class _Builder:
    """Accumulates events with monotonically increasing sequence numbers."""

    def __init__(self) -> None:
        self._seq = 0
        self.events: list[AgentEvent] = []

    def _add(self, event_type, name, **kw):
        event = AgentEvent(
            run_id=_RUN.id,
            sequence_number=self._seq,
            event_type=event_type,
            name=name,
            **kw,
        )
        self._seq += 1
        self.events.append(event)
        return event

    def success(self, op, *, completed_name=None, completed_input="__same__"):
        name, tool_input = op
        self._add(EventType.TOOL_CALL_STARTED, name, input=dict(tool_input))
        c_input = dict(tool_input) if completed_input == "__same__" else completed_input
        self._add(
            EventType.TOOL_CALL_COMPLETED,
            completed_name or name,
            input=c_input,
            output={"ok": True},
            status="ok",
        )
        return self

    def started(self, op):
        name, tool_input = op
        self._add(EventType.TOOL_CALL_STARTED, name, input=dict(tool_input))
        return self

    def completed(self, op):
        name, tool_input = op
        self._add(EventType.TOOL_CALL_COMPLETED, name, input=dict(tool_input), output={"ok": True})
        return self

    def error(self, name="temporary_failure", **kw):
        self._add(EventType.ERROR, name, **kw)
        return self

    def decision(self, name="decide", **kw):
        self._add(EventType.DECISION, name, **kw)
        return self

    def llm_completed(self, name="llm"):
        self._add(EventType.LLM_CALL_COMPLETED, name, output={"text": "hi"})
        return self

    def lifecycle_start(self):
        self._add(EventType.RUN_STARTED, "run_started")
        return self

    def lifecycle_end(self):
        self._add(EventType.RUN_COMPLETED, "run_completed")
        return self


def _detect(events):
    return DuplicateToolDetector().detect(_RUN, events)


# ---------------------------------------------------------------------------
# A. Public API
# ---------------------------------------------------------------------------


def test_importable_from_package_root():
    from agentlens.detectors import DuplicateToolDetector as Imported

    assert Imported is DuplicateToolDetector


def test_satisfies_issue_detector_protocol():
    assert isinstance(DuplicateToolDetector(), IssueDetector)


def test_detect_returns_list():
    assert DuplicateToolDetector().detect(_RUN, []) == []


def test_structural_protocol_accepts_any_detect_impl():
    class Custom:
        def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
            return []

    assert isinstance(Custom(), IssueDetector)


# ---------------------------------------------------------------------------
# B. Enum alias
# ---------------------------------------------------------------------------


def test_duplicate_alias_identity_and_value():
    assert IssueType.DUPLICATE is IssueType.DUPLICATE_TOOL_CALL
    assert IssueType.DUPLICATE.value == "duplicate_tool_call"


def test_enum_iteration_unchanged():
    assert [m.name for m in IssueType] == [
        "AGENT_LOOP",
        "EXCESSIVE_RETRY",
        "DUPLICATE_TOOL_CALL",
        "TOOL_FAILURE",
        "INEFFICIENCY",
    ]


# ---------------------------------------------------------------------------
# C. Core positive detection
# ---------------------------------------------------------------------------


def test_two_identical_successful_calls_produce_one_issue():
    events = _Builder().success(_OP_A).success(_OP_A).events
    issues = _detect(events)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type is IssueType.DUPLICATE
    assert issue.severity is Severity.MEDIUM
    assert issue.run_id == _RUN.id
    assert issue.related_event_ids == [e.id for e in events]  # start, complete, start, complete
    assert len(issue.related_event_ids) == 4


def test_first_call_is_not_flagged_alone():
    events = _Builder().success(_OP_A).events
    assert _detect(events) == []


def test_metadata_contents():
    b = _Builder().success(_OP_A).decision("recheck").success(_OP_A)
    issue = _detect(b.events)[0]
    assert issue.metadata == {
        "detector": "duplicate_tool_detector",
        "operation": "lookup_customer",
        "input": {"customer_id": "123"},
        "original_sequence_number": 0,
        "duplicate_sequence_number": 3,
        "duplicate_count": 1,
    }


def test_related_ids_reference_original_and_duplicate_pairs_in_trace_order():
    b = _Builder()
    b.success(_OP_A)  # seq 0,1
    b.decision("x")  # seq 2
    b.success(_OP_A)  # seq 3,4
    events = b.events
    issue = _detect(events)[0]
    assert issue.related_event_ids == [events[0].id, events[1].id, events[3].id, events[4].id]


# ---------------------------------------------------------------------------
# D. Multiple duplicates
# ---------------------------------------------------------------------------


def test_three_identical_calls_produce_two_issues_escalating_severity():
    b = _Builder().success(_OP_A).success(_OP_A).success(_OP_A)
    issues = _detect(b.events)
    assert [i.severity for i in issues] == [Severity.MEDIUM, Severity.HIGH]
    assert [i.metadata["duplicate_count"] for i in issues] == [1, 2]
    # every issue references the same original pair
    original_ids = {tuple(i.related_event_ids[:2]) for i in issues}
    assert len(original_ids) == 1


def test_five_identical_calls_escalate_to_critical():
    b = _Builder()
    for _ in range(5):
        b.success(_OP_A)
    issues = _detect(b.events)
    assert [i.severity for i in issues] == [
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.HIGH,
        Severity.CRITICAL,
    ]
    for issue in issues:
        assert len(issue.related_event_ids) == len(set(issue.related_event_ids)) == 4


# ---------------------------------------------------------------------------
# E. Operation identity
# ---------------------------------------------------------------------------


def test_same_name_same_input_is_duplicate():
    assert len(_detect(_Builder().success(_OP_A).success(_OP_A).events)) == 1


def test_same_name_different_input_is_not_duplicate():
    b = _Builder().success(("t", {"id": "1"})).success(("t", {"id": "2"}))
    assert _detect(b.events) == []


def test_different_name_same_input_is_not_duplicate():
    b = _Builder().success(("lookup", {"id": "1"})).success(("fetch", {"id": "1"}))
    assert _detect(b.events) == []


def test_flat_dict_key_order_is_equivalent():
    b = _Builder().success(("t", {"a": 1, "b": 2})).success(("t", {"b": 2, "a": 1}))
    assert len(_detect(b.events)) == 1


def test_nested_dict_key_order_is_equivalent():
    left = {"outer": {"x": 1, "y": {"p": 1, "q": 2}}}
    right = {"outer": {"y": {"q": 2, "p": 1}, "x": 1}}
    b = _Builder().success(("t", left)).success(("t", right))
    assert len(_detect(b.events)) == 1


def test_list_order_is_significant():
    b = _Builder().success(("t", {"ids": [1, 2]})).success(("t", {"ids": [2, 1]}))
    assert _detect(b.events) == []


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"v": 1}, {"v": 1.0}),
        ({"v": 1}, {"v": "1"}),
        ({"v": True}, {"v": 1}),
        ({"v": None}, {"v": "None"}),
    ],
)
def test_primitive_distinctions_are_preserved(first, second):
    b = _Builder().success(("t", first)).success(("t", second))
    assert _detect(b.events) == []


# ---------------------------------------------------------------------------
# F. Successful pairing
# ---------------------------------------------------------------------------


def test_start_completed_is_a_valid_pair():
    assert len(_detect(_Builder().success(_OP_A).success(_OP_A).events)) == 1


def test_start_decision_completed_is_not_a_pair():
    b = _Builder()
    b.success(_OP_A)
    b.started(_OP_A)
    b.decision("interrupt")
    b.completed(_OP_A)
    assert _detect(b.events) == []


def test_start_error_is_not_a_success():
    b = _Builder()
    b.success(_OP_A)
    b.started(_OP_A)
    b.error()
    b.completed(_OP_A)  # orphan completion, no active start
    assert _detect(b.events) == []


def test_start_start_completed_only_pairs_the_second_start():
    b = _Builder()
    b.started(_OP_A)  # dropped
    b.started(_OP_A)  # paired
    b.completed(_OP_A)
    assert _detect(b.events) == []  # only one successful pair


def test_completed_name_mismatch_is_conservative():
    b = _Builder()
    b.success(_OP_A)
    b.success(_OP_A, completed_name="something_else")
    assert _detect(b.events) == []  # second pair invalid -> not a duplicate


def test_completed_input_mismatch_is_conservative():
    b = _Builder()
    b.success(_OP_A)
    b.success(_OP_A, completed_input={"customer_id": "999"})
    assert _detect(b.events) == []


def test_unresolved_starts_do_not_produce_issues():
    b = _Builder()
    b.started(_OP_A)
    b.started(_OP_A)
    b.started(_OP_A)
    assert _detect(b.events) == []


# ---------------------------------------------------------------------------
# G. Information boundaries
# ---------------------------------------------------------------------------


def test_repeat_with_no_new_information_is_a_duplicate():
    b = _Builder().success(_OP_A).decision("plain").success(_OP_A)
    assert len(_detect(b.events)) == 1


def test_decision_uses_new_information_false_does_not_reset_history():
    b = _Builder()
    b.success(_OP_A)
    b.decision("recheck", output={"uses_new_information": False}, metadata={"justified": False})
    b.success(_OP_A)
    assert len(_detect(b.events)) == 1


def test_decision_depends_on_new_information_false_does_not_reset_history():
    b = _Builder()
    b.success(_OP_A)
    b.decision("recheck", metadata={"depends_on_new_information": False})
    b.success(_OP_A)
    assert len(_detect(b.events)) == 1


def test_decision_without_metadata_does_not_reset_history():
    b = _Builder().success(_OP_A).decision("think").decision("think again").success(_OP_A)
    assert len(_detect(b.events)) == 1


def test_successful_different_tool_resets_information_generation():
    b = _Builder().success(_OP_A).success(_OP_B).success(_OP_A)
    assert _detect(b.events) == []


def test_llm_call_completed_resets_information_generation():
    b = _Builder().success(_OP_A).llm_completed().success(_OP_A)
    assert _detect(b.events) == []


def test_explicit_new_information_event_resets_history():
    b = _Builder()
    b.success(_OP_A)
    b.decision("refresh", metadata={"uses_new_information": True})
    b.success(_OP_A)
    assert _detect(b.events) == []


def test_lifecycle_events_do_not_create_or_justify_duplicates():
    b = _Builder()
    b.lifecycle_start()
    b.success(_OP_A)
    b.lifecycle_end()
    b.lifecycle_start()
    b.success(_OP_A)
    b.lifecycle_end()
    # lifecycle events are ignored entirely: the two successes are still a duplicate
    assert len(_detect(b.events)) == 1


# ---------------------------------------------------------------------------
# H. False positives
# ---------------------------------------------------------------------------


def test_repeated_decisions_only():
    b = _Builder()
    for _ in range(5):
        b.decision("analyze")
    assert _detect(b.events) == []


def test_retry_then_success_is_not_a_duplicate():
    b = _Builder()
    b.started(_OP_A)
    b.error()
    b.started(_OP_A)
    b.completed(_OP_A)
    assert _detect(b.events) == []


def test_failed_attempts_only():
    b = _Builder()
    for _ in range(4):
        b.started(_OP_A)
        b.error()
    assert _detect(b.events) == []


def test_different_inputs_are_not_duplicates():
    b = (
        _Builder()
        .success(("t", {"id": "1"}))
        .success(("t", {"id": "2"}))
        .success(("t", {"id": "3"}))
    )
    assert _detect(b.events) == []


def test_calls_separated_by_genuine_new_information():
    b = _Builder().success(_OP_A).success(_OP_B).success(_OP_A).success(_OP_B).success(_OP_A)
    assert _detect(b.events) == []


# ---------------------------------------------------------------------------
# I. Ordering and purity
# ---------------------------------------------------------------------------


def test_shuffled_input_is_detected_by_sequence_number():
    events = _Builder().success(_OP_A).decision("x").success(_OP_A).events
    shuffled = [events[4], events[0], events[2], events[3], events[1]]
    issues = DuplicateToolDetector().detect(_RUN, shuffled)
    assert len(issues) == 1
    assert issues[0].related_event_ids == [events[0].id, events[1].id, events[3].id, events[4].id]


def test_caller_list_is_not_reordered():
    events = _Builder().success(_OP_A).success(_OP_A).events
    shuffled = [events[3], events[0], events[2], events[1]]
    snapshot = list(shuffled)
    DuplicateToolDetector().detect(_RUN, shuffled)
    assert shuffled == snapshot


def test_detection_does_not_mutate_events_or_nested_payloads():
    nested = {"filters": {"tags": ["x", "y"]}}
    b = _Builder()
    b.started(("t", deepcopy(nested)))
    b._add(EventType.TOOL_CALL_COMPLETED, "t", input=deepcopy(nested), output={"rows": [1, 2]})
    b.started(("t", deepcopy(nested)))
    b._add(EventType.TOOL_CALL_COMPLETED, "t", input=deepcopy(nested), output={"rows": [1, 2]})
    events = b.events

    before = [e.model_dump() for e in events]
    DuplicateToolDetector().detect(_RUN, events)
    assert [e.model_dump() for e in events] == before
    assert events[0].input == nested


def test_detection_does_not_mutate_the_run():
    run = AgentRun(task="unchanged")
    before = run.model_dump()
    DuplicateToolDetector().detect(run, _Builder().success(_OP_A).success(_OP_A).events)
    assert run.model_dump() == before


# ---------------------------------------------------------------------------
# J. AgentIssue validity
# ---------------------------------------------------------------------------


def test_issue_shape_is_valid():
    issue = _detect(_Builder().success(_OP_A).success(_OP_A).events)[0]
    assert isinstance(issue, AgentIssue)
    assert issue.run_id == _RUN.id
    assert issue.issue_type is IssueType.DUPLICATE
    assert issue.issue_type is IssueType.DUPLICATE_TOOL_CALL
    assert issue.issue_type.value == "duplicate_tool_call"
    assert isinstance(issue.severity, Severity)
    assert all(isinstance(eid, UUID) for eid in issue.related_event_ids)
    assert len(issue.related_event_ids) == len(set(issue.related_event_ids))
    assert issue.description


def test_metadata_json_round_trips():
    issue = _detect(_Builder().success(_OP_A).success(_OP_A).events)[0]
    assert json.loads(json.dumps(issue.metadata)) == issue.metadata
    restored = AgentIssue.model_validate_json(issue.model_dump_json())
    assert restored == issue


# ---------------------------------------------------------------------------
# K. Determinism
# ---------------------------------------------------------------------------


def test_repeated_detection_is_content_equivalent():
    events = _Builder().success(_OP_A).success(_OP_A).success(_OP_A).events
    first = _detect(events)
    second = _detect(list(reversed(events)))
    assert len(first) == len(second) == 2
    for a, b in zip(first, second, strict=True):
        assert a.metadata == b.metadata
        assert a.related_event_ids == b.related_event_ids
        assert a.severity == b.severity
        assert a.description == b.description

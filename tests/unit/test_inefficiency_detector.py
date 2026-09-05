"""Synthetic unit tests for :class:`agentlens.detectors.InefficiencyDetector`."""

import json
from collections.abc import Sequence
from copy import deepcopy
from uuid import UUID

import pytest

from agentlens.detectors import InefficiencyDetector, IssueDetector
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

_RUN = AgentRun(task="synthetic trace")


class _Builder:
    def __init__(self) -> None:
        self._seq = 0
        self.events: list[AgentEvent] = []

    def _add(self, event_type, name, **kw):
        event = AgentEvent(
            run_id=_RUN.id, sequence_number=self._seq, event_type=event_type, name=name, **kw
        )
        self._seq += 1
        self.events.append(event)
        return event

    def op(
        self,
        name,
        tool_input=None,
        *,
        started_meta=None,
        completed_meta=None,
        completed_output=None,
    ):
        self._add(EventType.TOOL_CALL_STARTED, name, input=tool_input, metadata=started_meta or {})
        self._add(
            EventType.TOOL_CALL_COMPLETED,
            name,
            input=tool_input,
            output=completed_output if completed_output is not None else {"ok": True},
            status="ok",
            metadata=completed_meta or {},
        )
        return self

    def started(self, name, tool_input=None, **kw):
        self._add(EventType.TOOL_CALL_STARTED, name, input=tool_input, **kw)
        return self

    def completed(self, name, tool_input=None, **kw):
        self._add(EventType.TOOL_CALL_COMPLETED, name, input=tool_input, **kw)
        return self

    def error(self, name="boom", **kw):
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
    return InefficiencyDetector().detect(_RUN, events)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_importable_from_package_root():
    from agentlens.detectors import InefficiencyDetector as Imported

    assert Imported is InefficiencyDetector


def test_satisfies_issue_detector_protocol():
    assert isinstance(InefficiencyDetector(), IssueDetector)


def test_detect_returns_list():
    assert InefficiencyDetector().detect(_RUN, []) == []


def test_structural_protocol_accepts_any_detect_impl():
    class Custom:
        def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
            return []

    assert isinstance(Custom(), IssueDetector)


# ---------------------------------------------------------------------------
# Positive detection
# ---------------------------------------------------------------------------


def test_scope_mismatch_alone_is_medium():
    b = _Builder().op(
        "broad",
        {},
        started_meta={"required_scope": "one", "actual_scope": "all"},
    )
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].severity is Severity.MEDIUM
    assert issues[0].metadata["evidence"] == ["scope_mismatch"]
    assert issues[0].metadata["operation"] == "broad"


def test_wasted_work_true_on_completed_is_flagged():
    b = _Builder().op("t", {"x": 1}, completed_meta={"wasted_work": True})
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].metadata["evidence"] == ["explicit_wasted_work"]
    assert issues[0].severity is Severity.MEDIUM


def test_necessary_false_on_operation_is_flagged():
    b = _Builder().op("t", {}, completed_meta={"necessary": False})
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].metadata["evidence"] == ["explicit_unnecessary"]


def test_later_wasted_operation_recognition_is_flagged():
    b = _Builder()
    b.op("fetch_all", {})
    b.decision("realize", metadata={"wasted_operation": "fetch_all"})
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].metadata["evidence"] == ["explicit_wasted_operation"]
    # related includes the two op events plus the recognition event
    assert len(issues[0].related_event_ids) == 3


def test_multiple_signals_increase_severity_to_high():
    b = _Builder()
    b.op("first", {}, completed_meta={"sufficient_to_answer": True})
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].severity is Severity.HIGH
    assert issues[0].metadata["evidence"] == [
        "scope_mismatch",
        "sufficient_information_already_available",
    ]


def test_three_signals_is_critical():
    b = _Builder()
    b.op("first", {}, completed_meta={"sufficient_to_answer": True})
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    b.decision("realize", metadata={"wasted_operation": "broad"})
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].severity is Severity.CRITICAL
    assert issues[0].metadata["evidence"] == [
        "scope_mismatch",
        "explicit_wasted_operation",
        "sufficient_information_already_available",
    ]


def test_same_operation_with_repeated_signals_produces_one_issue():
    b = _Builder()
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    b.decision("r1", metadata={"wasted_operation": "broad"})
    b.decision("r2", metadata={"wasted_operation": "broad"})
    b.decision("r3", output={"wasted_operation": "broad"})
    issues = _detect(b.events)
    assert len(issues) == 1


def test_multiple_distinct_inefficient_operations_in_trace_order():
    b = _Builder()
    b.op("alpha", {}, completed_meta={"wasted_work": True})
    b.decision("gap")
    b.op("beta", {}, started_meta={"required_scope": "x", "actual_scope": "y"})
    issues = _detect(b.events)
    assert [i.metadata["operation"] for i in issues] == ["alpha", "beta"]
    assert issues[0].related_event_ids[0] != issues[1].related_event_ids[0]


# ---------------------------------------------------------------------------
# Required-scope behaviour
# ---------------------------------------------------------------------------


def test_equal_scopes_do_not_trigger():
    b = _Builder().op("t", {}, started_meta={"required_scope": "one", "actual_scope": "one"})
    assert _detect(b.events) == []


def test_missing_actual_scope_does_not_trigger():
    b = _Builder().op("t", {}, started_meta={"required_scope": "one", "chosen_scope": "all"})
    assert _detect(b.events) == []


def test_missing_required_scope_does_not_trigger():
    b = _Builder().op("t", {}, started_meta={"actual_scope": "all"})
    assert _detect(b.events) == []


def test_non_string_scope_values_are_handled():
    # equal ints -> no mismatch
    assert (
        _detect(
            _Builder().op("t", {}, started_meta={"required_scope": 1, "actual_scope": 1}).events
        )
        == []
    )
    # unequal ints -> mismatch
    hit = _detect(
        _Builder().op("t", {}, started_meta={"required_scope": 1, "actual_scope": 2}).events
    )
    assert len(hit) == 1
    # different types -> conservatively no mismatch
    assert (
        _detect(
            _Builder().op("t", {}, started_meta={"required_scope": "1", "actual_scope": 1}).events
        )
        == []
    )
    # null values -> no mismatch
    assert (
        _detect(
            _Builder()
            .op("t", {}, started_meta={"required_scope": None, "actual_scope": "all"})
            .events
        )
        == []
    )
    # list values compare without crashing
    assert (
        len(
            _detect(
                _Builder()
                .op("t", {}, started_meta={"required_scope": [1], "actual_scope": [2]})
                .events
            )
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Sufficient-information behaviour
# ---------------------------------------------------------------------------


def test_sufficient_true_is_recognized_but_not_alone_triggering():
    b = _Builder()
    b.op("first", {}, completed_meta={"sufficient_to_answer": True})
    b.op("second", {})  # identical-ish work but no explicit waste evidence
    assert _detect(b.events) == []


@pytest.mark.parametrize("value", [False, "true", "True", 1, 0, None])
def test_non_true_sufficiency_values_do_not_count(value):
    b = _Builder()
    b.op("first", {}, completed_meta={"sufficient_to_answer": value})
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    issues = _detect(b.events)
    assert issues[0].metadata["evidence"] == ["scope_mismatch"]  # sufficiency NOT counted


def test_sufficiency_read_from_mapping_output():
    b = _Builder()
    b.op("first", {}, completed_output={"sufficient_to_answer": True})
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    assert "sufficient_information_already_available" in _detect(b.events)[0].metadata["evidence"]


# ---------------------------------------------------------------------------
# Explicit recognition association
# ---------------------------------------------------------------------------


def test_wasted_operation_binds_to_most_recent_preceding_same_name_op():
    b = _Builder()
    b.op("fetch", {"n": 1})  # ops[0]
    b.decision("noise")
    b.op("fetch", {"n": 2})  # ops[1] - most recent preceding for the recognition
    b.decision("realize", metadata={"wasted_operation": "fetch"})
    issues = _detect(b.events)
    assert len(issues) == 1
    assert issues[0].metadata["input"] == {"n": 2}


def test_no_matching_prior_operation_does_not_crash_or_emit():
    b = _Builder()
    b.decision("realize", metadata={"wasted_operation": "never_called"})
    b.op("something_else", {})
    assert _detect(b.events) == []


def test_repeated_recognition_events_do_not_duplicate_issue():
    b = _Builder()
    b.op("fetch", {})
    b.decision("r1", metadata={"wasted_operation": "fetch"})
    b.decision("r2", metadata={"wasted_operation": "fetch"})
    assert len(_detect(b.events)) == 1


# ---------------------------------------------------------------------------
# Tool pairing
# ---------------------------------------------------------------------------


def test_valid_start_completed_pair():
    b = _Builder().op("t", {}, completed_meta={"wasted_work": True})
    assert len(_detect(b.events)) == 1


def test_start_error_is_not_successful():
    b = _Builder()
    b.started("t", {}, metadata={"wasted_work": True})
    b.error()
    b.completed("t", {}, metadata={"wasted_work": True})  # orphan completion
    assert _detect(b.events) == []


def test_interrupted_pair_is_not_successful():
    b = _Builder()
    b.started("t", {}, metadata={"wasted_work": True})
    b.decision("interrupt")
    b.completed("t", {}, output={"ok": True})
    assert _detect(b.events) == []


def test_completion_name_mismatch_is_conservative():
    b = _Builder()
    b.started("t", {})
    b.completed("other", {}, metadata={"wasted_work": True})
    assert _detect(b.events) == []


def test_completion_input_mismatch_is_conservative():
    b = _Builder()
    b.started("t", {"a": 1})
    b.completed("t", {"a": 2}, metadata={"wasted_work": True})
    assert _detect(b.events) == []


def test_unresolved_start_produces_no_issue():
    b = _Builder()
    b.started("t", {}, metadata={"wasted_work": True})
    assert _detect(b.events) == []


# ---------------------------------------------------------------------------
# False positives
# ---------------------------------------------------------------------------


def test_repeated_successful_calls_without_waste_evidence():
    b = _Builder().op("t", {"x": 1}).op("t", {"x": 1}).op("t", {"x": 1})
    assert _detect(b.events) == []


def test_multiple_unrelated_tool_calls():
    b = _Builder().op("a", {"i": 1}).op("b", {"i": 2}).op("c", {"i": 3})
    assert _detect(b.events) == []


def test_long_duration_is_irrelevant():
    b = _Builder()
    b._add(EventType.TOOL_CALL_STARTED, "slow", input={})
    b._add(
        EventType.TOOL_CALL_COMPLETED, "slow", input={}, output={"ok": True}, duration_ms=999999.0
    )
    assert _detect(b.events) == []


def test_lifecycle_only_trace():
    b = _Builder().lifecycle_start().lifecycle_end()
    assert _detect(b.events) == []


def test_decisions_only_trace():
    b = _Builder()
    for _ in range(4):
        b.decision("think", metadata={"necessary": False, "wasted_work": True})
    assert _detect(b.events) == []


def test_errors_only_trace():
    b = _Builder().error().error().error()
    assert _detect(b.events) == []


def test_many_records_returned_without_scope_metadata_is_not_flagged():
    b = _Builder().op("fetch_all", {}, completed_output={"rows": list(range(1000))})
    assert _detect(b.events) == []


# ---------------------------------------------------------------------------
# Ordering and purity
# ---------------------------------------------------------------------------


def test_shuffled_input_is_analysed_by_sequence_number():
    b = _Builder()
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    b.decision("realize", metadata={"wasted_operation": "broad"})
    events = b.events
    shuffled = [events[2], events[0], events[1]]
    issues = InefficiencyDetector().detect(_RUN, shuffled)
    assert len(issues) == 1
    assert issues[0].related_event_ids == [events[0].id, events[1].id, events[2].id]


def test_caller_list_is_not_reordered():
    b = _Builder().op("broad", {}, started_meta={"required_scope": "a", "actual_scope": "b"})
    events = b.events
    shuffled = [events[1], events[0]]
    snapshot = list(shuffled)
    InefficiencyDetector().detect(_RUN, shuffled)
    assert shuffled == snapshot


def test_detection_does_not_mutate_events_or_payloads():
    nested = {"required_scope": "one", "actual_scope": "all", "extra": {"k": [1, 2]}}
    b = _Builder()
    b.started("broad", {"filter": {"tags": ["x"]}}, metadata=deepcopy(nested))
    b.completed("broad", {"filter": {"tags": ["x"]}}, output={"ok": True})
    events = b.events
    before = [e.model_dump() for e in events]
    InefficiencyDetector().detect(_RUN, events)
    assert [e.model_dump() for e in events] == before


def test_detection_does_not_mutate_run():
    run = AgentRun(task="unchanged")
    before = run.model_dump()
    b = _Builder().op("broad", {}, started_meta={"required_scope": "a", "actual_scope": "b"})
    InefficiencyDetector().detect(run, b.events)
    assert run.model_dump() == before


def test_repeated_calls_produce_equivalent_content():
    b = _Builder()
    b.op("first", {}, completed_meta={"sufficient_to_answer": True})
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    b.decision("realize", metadata={"wasted_operation": "broad"})
    first = _detect(b.events)[0]
    second = _detect(list(reversed(b.events)))[0]
    assert first.metadata == second.metadata
    assert first.related_event_ids == second.related_event_ids
    assert first.severity == second.severity
    assert first.description == second.description


# ---------------------------------------------------------------------------
# AgentIssue validity
# ---------------------------------------------------------------------------


def test_issue_shape_is_valid():
    b = _Builder().op(
        "broad", {"k": "v"}, started_meta={"required_scope": "a", "actual_scope": "b"}
    )
    issue = _detect(b.events)[0]
    assert isinstance(issue, AgentIssue)
    assert issue.run_id == _RUN.id
    assert issue.issue_type is IssueType.INEFFICIENCY
    assert issue.issue_type.value == "inefficiency"
    assert isinstance(issue.severity, Severity)
    assert all(isinstance(eid, UUID) for eid in issue.related_event_ids)
    assert len(issue.related_event_ids) == len(set(issue.related_event_ids))
    assert issue.description


def test_related_ids_in_trace_order_no_lifecycle():
    b = _Builder()
    b.lifecycle_start()
    b.op("first", {}, completed_meta={"sufficient_to_answer": True})
    b.op("broad", {}, started_meta={"required_scope": "one", "actual_scope": "all"})
    b.decision("realize", metadata={"wasted_operation": "broad"})
    b.lifecycle_end()
    events = b.events
    issue = _detect(events)[0]

    lifecycle_ids = {
        e.id for e in events if e.event_type in {EventType.RUN_STARTED, EventType.RUN_COMPLETED}
    }
    assert lifecycle_ids.isdisjoint(issue.related_event_ids)
    by_id = {e.id: e for e in events}
    seqs = [by_id[i].sequence_number for i in issue.related_event_ids]
    assert seqs == sorted(seqs)


def test_metadata_json_round_trips():
    b = _Builder().op(
        "broad", {"k": "v"}, started_meta={"required_scope": "a", "actual_scope": "b"}
    )
    issue = _detect(b.events)[0]
    assert json.loads(json.dumps(issue.metadata)) == issue.metadata
    restored = AgentIssue.model_validate_json(issue.model_dump_json())
    assert restored == issue
    assert issue.metadata["input"] == {"k": "v"}

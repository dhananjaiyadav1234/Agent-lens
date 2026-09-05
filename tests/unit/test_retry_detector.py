"""Synthetic unit tests for :class:`agentlens.detectors.RetryDetector`.

Events / runs are built directly as universal models so arbitrary sequences can
be exercised. The detector under test still only ever sees universal models.
"""

import json
from collections.abc import Sequence
from copy import deepcopy
from uuid import UUID

import pytest

from agentlens.detectors import IssueDetector, RetryDetector
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

_RUN = AgentRun(task="synthetic trace")

_OP_A = ("fetch_customer", {"customer_id": "123"})
_OP_B = ("fetch_order", {"order_id": "999"})


def _ev(seq, event_type, name, *, input=None, output=None, metadata=None):
    return AgentEvent(
        run_id=_RUN.id,
        sequence_number=seq,
        event_type=event_type,
        name=name,
        input=input,
        output=output,
        metadata=metadata or {},
    )


def _started(seq, op):
    name, tool_input = op
    return _ev(seq, EventType.TOOL_CALL_STARTED, name, input=dict(tool_input))


def _error(seq, *, operation=None, extra=None):
    output = {}
    if operation is not None:
        output["operation"] = operation
    if extra:
        output.update(extra)
    return _ev(seq, EventType.ERROR, "temporary_tool_failure", output=output or None, metadata={})


def _completed(seq, op):
    name, tool_input = op
    return _ev(
        seq, EventType.TOOL_CALL_COMPLETED, name, input=dict(tool_input), output={"ok": True}
    )


def _decision(seq, name="decide"):
    return _ev(seq, EventType.DECISION, name)


def _failed_attempts(op, count, *, start=0, operation_in_error="__match__"):
    """`count` consecutive failed attempts of `op`, as a flat event list."""
    events = []
    seq = start
    op_name = op[0] if operation_in_error == "__match__" else operation_in_error
    for _ in range(count):
        events.append(_started(seq, op))
        events.append(_error(seq + 1, operation=None if operation_in_error is None else op_name))
        seq += 2
    return events


def _detect(events, **kwargs):
    return RetryDetector(**kwargs).detect(_RUN, events)


# ---------------------------------------------------------------------------
# A. Public API / interface
# ---------------------------------------------------------------------------


def test_importable_from_package_root():
    from agentlens.detectors import RetryDetector as Imported

    assert Imported is RetryDetector


def test_satisfies_issue_detector_protocol():
    assert isinstance(RetryDetector(), IssueDetector)


def test_detect_returns_list_of_agent_issues():
    result = RetryDetector().detect(_RUN, [])
    assert isinstance(result, list)
    assert result == []

    issues = _detect(_failed_attempts(_OP_A, 3))
    assert isinstance(issues, list)
    assert all(isinstance(i, AgentIssue) for i in issues)


def test_structural_protocol_accepts_any_detect_impl():
    class Custom:
        def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
            return []

    assert isinstance(Custom(), IssueDetector)


# ---------------------------------------------------------------------------
# B. Configuration
# ---------------------------------------------------------------------------


def test_default_minimum_is_three():
    assert RetryDetector().minimum_failed_attempts == 3


@pytest.mark.parametrize("value", [2, 3, 4, 10])
def test_valid_minimum_values_accepted(value):
    assert RetryDetector(minimum_failed_attempts=value).minimum_failed_attempts == value


@pytest.mark.parametrize("value", [1, 0, -1, -9])
def test_minimum_below_two_raises_value_error(value):
    with pytest.raises(ValueError, match="at least 2"):
        RetryDetector(minimum_failed_attempts=value)


@pytest.mark.parametrize("value", ["3", "abc", 3.0, 2.5, None, True, False, [3]])
def test_non_int_minimum_raises_type_error(value):
    with pytest.raises(TypeError, match="must be an int"):
        RetryDetector(minimum_failed_attempts=value)


def test_string_three_is_not_coerced():
    with pytest.raises(TypeError):
        RetryDetector(minimum_failed_attempts="3")


# ---------------------------------------------------------------------------
# C. Core detection
# ---------------------------------------------------------------------------


def test_three_failed_attempts_produce_one_issue():
    issues = _detect(_failed_attempts(_OP_A, 3))
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 3
    assert len(issues[0].related_event_ids) == 6


def test_below_threshold_produces_no_issue():
    assert _detect(_failed_attempts(_OP_A, 2)) == []


def test_exact_threshold_two_with_config():
    issues = _detect(_failed_attempts(_OP_A, 2), minimum_failed_attempts=2)
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 2
    assert issues[0].severity is Severity.MEDIUM


def test_above_threshold_counts_all_failures():
    issues = _detect(_failed_attempts(_OP_A, 5))
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 5
    assert len(issues[0].related_event_ids) == 10


def test_eventual_success_still_reports_preceding_failures():
    events = _failed_attempts(_OP_A, 3)
    events.append(_started(6, _OP_A))
    events.append(_completed(7, _OP_A))

    issues = _detect(events)
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 3
    # the successful final attempt is not referenced
    assert len(issues[0].related_event_ids) == 6
    assert events[6].id not in issues[0].related_event_ids
    assert events[7].id not in issues[0].related_event_ids


def test_sequence_ending_immediately_after_final_failed_attempt():
    events = _failed_attempts(_OP_A, 3)  # ends on an ERROR, nothing after
    issues = RetryDetector().detect(_RUN, events)
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 3
    assert issues[0].related_event_ids == [e.id for e in events]


def test_one_failed_attempt_is_never_an_issue():
    assert _detect(_failed_attempts(_OP_A, 1)) == []
    assert _detect(_failed_attempts(_OP_A, 1), minimum_failed_attempts=2) == []


# ---------------------------------------------------------------------------
# D. Operation identity
# ---------------------------------------------------------------------------


def test_same_name_same_input_is_one_operation():
    issues = _detect(_failed_attempts(_OP_A, 3))
    assert len(issues) == 1


def test_same_name_different_input_breaks_streak():
    events = [
        _started(0, ("fetch", {"id": "1"})),
        _error(1, operation="fetch"),
        _started(2, ("fetch", {"id": "2"})),
        _error(3, operation="fetch"),
        _started(4, ("fetch", {"id": "3"})),
        _error(5, operation="fetch"),
    ]
    assert _detect(events) == []


def test_different_name_same_input_breaks_streak():
    events = [
        _started(0, ("fetch", {"id": "1"})),
        _error(1, operation="fetch"),
        _started(2, ("lookup", {"id": "1"})),
        _error(3, operation="lookup"),
        _started(4, ("fetch", {"id": "1"})),
        _error(5, operation="fetch"),
    ]
    assert _detect(events) == []


def test_dict_key_order_does_not_affect_operation_identity():
    events = [
        _ev(0, EventType.TOOL_CALL_STARTED, "fetch", input={"a": 1, "b": 2}),
        _error(1),
        _ev(2, EventType.TOOL_CALL_STARTED, "fetch", input={"b": 2, "a": 1}),
        _error(3),
        _ev(4, EventType.TOOL_CALL_STARTED, "fetch", input={"a": 1, "b": 2}),
        _error(5),
    ]
    issues = _detect(events)
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 3


def test_nested_dict_key_order_does_not_affect_identity():
    left = {"outer": {"x": 1, "y": {"p": 1, "q": 2}}}
    right = {"outer": {"y": {"q": 2, "p": 1}, "x": 1}}
    events = [
        _ev(0, EventType.TOOL_CALL_STARTED, "fetch", input=deepcopy(left)),
        _error(1),
        _ev(2, EventType.TOOL_CALL_STARTED, "fetch", input=deepcopy(right)),
        _error(3),
        _ev(4, EventType.TOOL_CALL_STARTED, "fetch", input=deepcopy(left)),
        _error(5),
    ]
    assert len(_detect(events)) == 1


def test_list_order_matters_for_operation_identity():
    events = [
        _ev(0, EventType.TOOL_CALL_STARTED, "fetch", input={"ids": [1, 2]}),
        _error(1),
        _ev(2, EventType.TOOL_CALL_STARTED, "fetch", input={"ids": [2, 1]}),
        _error(3),
        _ev(4, EventType.TOOL_CALL_STARTED, "fetch", input={"ids": [1, 2]}),
        _error(5),
    ]
    assert _detect(events) == []


def test_primitive_values_remain_distinct():
    events = [
        _ev(0, EventType.TOOL_CALL_STARTED, "fetch", input={"v": 1}),
        _error(1),
        _ev(2, EventType.TOOL_CALL_STARTED, "fetch", input={"v": 1.0}),
        _error(3),
        _ev(4, EventType.TOOL_CALL_STARTED, "fetch", input={"v": "1"}),
        _error(5),
        _ev(6, EventType.TOOL_CALL_STARTED, "fetch", input={"v": True}),
        _error(7),
    ]
    assert _detect(events, minimum_failed_attempts=2) == []


# ---------------------------------------------------------------------------
# E. Error association
# ---------------------------------------------------------------------------


def test_error_with_matching_operation_counts():
    events = _failed_attempts(_OP_A, 3, operation_in_error="fetch_customer")
    assert len(_detect(events)) == 1


def test_error_with_no_operation_counts_when_adjacent():
    events = _failed_attempts(_OP_A, 3, operation_in_error=None)
    issues = _detect(events)
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 3


def test_error_with_mismatched_operation_breaks_streak():
    events = [
        _started(0, _OP_A),
        _error(1, operation="fetch_customer"),
        _started(2, _OP_A),
        _error(3, operation="something_else"),
        _started(4, _OP_A),
        _error(5, operation="fetch_customer"),
    ]
    assert _detect(events) == []


def test_error_output_non_mapping_is_handled_safely():
    events = [
        _ev(0, EventType.TOOL_CALL_STARTED, "fetch", input={"id": "1"}),
        _ev(1, EventType.ERROR, "boom", output="a plain string"),
        _ev(2, EventType.TOOL_CALL_STARTED, "fetch", input={"id": "1"}),
        _ev(3, EventType.ERROR, "boom", output=["a", "list"]),
        _ev(4, EventType.TOOL_CALL_STARTED, "fetch", input={"id": "1"}),
        _ev(5, EventType.ERROR, "boom", output=None),
    ]
    issues = _detect(events)
    assert len(issues) == 1
    assert issues[0].metadata["failed_attempts"] == 3


def test_error_operation_non_string_is_conservative():
    events = [
        _started(0, _OP_A),
        _ev(1, EventType.ERROR, "boom", output={"operation": 123}),
        _started(2, _OP_A),
        _ev(3, EventType.ERROR, "boom", output={"operation": 123}),
        _started(4, _OP_A),
        _ev(5, EventType.ERROR, "boom", output={"operation": 123}),
    ]
    assert _detect(events) == []


def test_malformed_error_payloads_do_not_crash():
    events = [
        _started(0, _OP_A),
        _ev(1, EventType.ERROR, "boom", output={"nested": {"deep": [1, {"x": None}]}}),
        _started(2, _OP_A),
        _ev(3, EventType.ERROR, "boom", output=42),
        _started(4, _OP_A),
        _ev(5, EventType.ERROR, "boom", output=True),
    ]
    issues = _detect(events)  # must not raise
    assert len(issues) == 1


# ---------------------------------------------------------------------------
# F. Boundary behaviour
# ---------------------------------------------------------------------------


def test_different_tool_operation_breaks_streak():
    events = [
        *_failed_attempts(_OP_A, 2),
        _started(4, _OP_B),
        _error(5, operation="fetch_order"),
        *_failed_attempts(_OP_A, 2, start=6),
    ]
    assert _detect(events) == []


def test_decision_breaks_streak():
    events = [
        *_failed_attempts(_OP_A, 2),
        _decision(4),
        *_failed_attempts(_OP_A, 2, start=5),
    ]
    assert _detect(events) == []


def test_unrelated_event_between_start_and_error_breaks_pairing():
    events = [
        _started(0, _OP_A),
        _ev(1, EventType.LLM_CALL_STARTED, "think"),
        _error(2, operation="fetch_customer"),
        _started(3, _OP_A),
        _error(4, operation="fetch_customer"),
        _started(5, _OP_A),
        _error(6, operation="fetch_customer"),
    ]
    # first attempt is not a clean START->ERROR pair, so only 2 failures accrue
    assert _detect(events) == []


def test_lifecycle_events_do_not_create_retries():
    events = [
        _ev(0, EventType.RUN_STARTED, "run_started"),
        _ev(1, EventType.RUN_COMPLETED, "run_completed"),
    ]
    assert _detect(events) == []


def test_retry_streak_does_not_span_a_lifecycle_boundary():
    events = [
        *_failed_attempts(_OP_A, 2),
        _ev(4, EventType.RUN_COMPLETED, "run_completed"),
        _ev(5, EventType.RUN_STARTED, "run_started"),
        *_failed_attempts(_OP_A, 2, start=6),
    ]
    assert _detect(events) == []


# ---------------------------------------------------------------------------
# G. False positives
# ---------------------------------------------------------------------------


def test_successful_duplicate_calls_are_not_retries():
    events = [
        _started(0, _OP_A),
        _completed(1, _OP_A),
        _decision(2),
        _started(3, _OP_A),
        _completed(4, _OP_A),
    ]
    assert _detect(events) == []


def test_repeated_decisions_are_not_retries():
    events = [_decision(i, "analyze") for i in range(6)]
    assert _detect(events) == []


def test_repeated_tool_calls_without_errors_are_not_retries():
    events = []
    seq = 0
    for _ in range(5):
        events.append(_started(seq, _OP_A))
        events.append(_completed(seq + 1, _OP_A))
        seq += 2
    assert _detect(events) == []


def test_alternating_operations_are_not_retries():
    events = []
    seq = 0
    for _ in range(4):
        events.append(_started(seq, _OP_A))
        events.append(_error(seq + 1, operation="fetch_customer"))
        events.append(_started(seq + 2, _OP_B))
        events.append(_error(seq + 3, operation="fetch_order"))
        seq += 4
    assert _detect(events) == []


def test_isolated_errors_without_preceding_start_are_ignored():
    events = [_error(i) for i in range(5)]
    assert _detect(events) == []


def test_errors_without_active_attempt_between_completed_calls():
    events = [
        _started(0, _OP_A),
        _completed(1, _OP_A),
        _error(2),
        _error(3),
        _error(4),
    ]
    assert _detect(events) == []


# ---------------------------------------------------------------------------
# H. Multiple regions
# ---------------------------------------------------------------------------


def test_two_independent_retry_regions_produce_two_issues_in_order():
    region_a = _failed_attempts(_OP_A, 3, start=0)
    decision = [_decision(6, "regroup")]
    region_b = _failed_attempts(_OP_B, 3, start=7)
    events = region_a + decision + region_b

    issues = _detect(events)
    assert len(issues) == 2

    first, second = issues
    assert first.metadata["operation"] == "fetch_customer"
    assert second.metadata["operation"] == "fetch_order"

    first_ids = set(first.related_event_ids)
    second_ids = set(second.related_event_ids)
    assert first_ids.isdisjoint(second_ids)
    assert first.related_event_ids == [e.id for e in region_a]
    assert second.related_event_ids == [e.id for e in region_b]


def test_different_operation_without_separator_still_splits_into_two_issues():
    events = _failed_attempts(_OP_A, 3, start=0) + _failed_attempts(_OP_B, 3, start=6)
    issues = _detect(events)
    assert [i.metadata["operation"] for i in issues] == ["fetch_customer", "fetch_order"]


# ---------------------------------------------------------------------------
# I. Severity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        (2, Severity.MEDIUM),
        (3, Severity.MEDIUM),
        (4, Severity.HIGH),
        (5, Severity.HIGH),
        (6, Severity.CRITICAL),
        (9, Severity.CRITICAL),
    ],
)
def test_severity_scales_with_failed_attempt_count(failures, expected):
    issues = _detect(_failed_attempts(_OP_A, failures), minimum_failed_attempts=2)
    assert len(issues) == 1
    assert issues[0].severity is expected
    assert issues[0].metadata["failed_attempts"] == failures


# ---------------------------------------------------------------------------
# J. Ordering and purity
# ---------------------------------------------------------------------------


def test_detection_uses_sequence_number_not_list_position():
    events = _failed_attempts(_OP_A, 3)
    shuffled = [events[4], events[1], events[5], events[0], events[3], events[2]]
    issues = RetryDetector().detect(_RUN, shuffled)
    assert len(issues) == 1
    assert issues[0].related_event_ids == [e.id for e in events]


def test_caller_list_is_not_reordered():
    events = _failed_attempts(_OP_A, 3)
    shuffled = [events[3], events[0], events[5], events[1], events[4], events[2]]
    snapshot = list(shuffled)
    RetryDetector().detect(_RUN, shuffled)
    assert shuffled == snapshot


def test_detection_does_not_mutate_events_or_nested_payloads():
    nested = {"filters": {"tags": ["x", "y"]}}
    events = [
        _ev(0, EventType.TOOL_CALL_STARTED, "fetch", input=deepcopy(nested)),
        _ev(1, EventType.ERROR, "boom", output={"operation": "fetch", "detail": {"code": 1}}),
        _ev(2, EventType.TOOL_CALL_STARTED, "fetch", input=deepcopy(nested)),
        _ev(3, EventType.ERROR, "boom", output={"operation": "fetch"}),
        _ev(4, EventType.TOOL_CALL_STARTED, "fetch", input=deepcopy(nested)),
        _ev(5, EventType.ERROR, "boom", output={"operation": "fetch"}),
    ]
    before = [e.model_dump() for e in events]
    RetryDetector().detect(_RUN, events)
    assert [e.model_dump() for e in events] == before
    assert events[0].input == nested


def test_detection_does_not_mutate_the_run():
    run = AgentRun(task="unchanged")
    before = run.model_dump()
    RetryDetector().detect(run, _failed_attempts(_OP_A, 3))
    assert run.model_dump() == before


def test_repeated_calls_produce_equivalent_content():
    events = _failed_attempts(_OP_A, 4)
    first = _detect(events)[0]
    second = _detect(list(reversed(events)))[0]
    assert first.metadata == second.metadata
    assert first.related_event_ids == second.related_event_ids
    assert first.description == second.description
    assert first.severity == second.severity


# ---------------------------------------------------------------------------
# K. AgentIssue validity
# ---------------------------------------------------------------------------


def test_issue_shape_is_valid():
    issue = _detect(_failed_attempts(_OP_A, 3))[0]
    assert isinstance(issue, AgentIssue)
    assert issue.run_id == _RUN.id
    assert issue.issue_type is IssueType.RETRY
    assert issue.issue_type is IssueType.EXCESSIVE_RETRY
    assert issue.issue_type.value == "excessive_retry"
    assert isinstance(issue.severity, Severity)
    assert all(isinstance(eid, UUID) for eid in issue.related_event_ids)
    assert len(issue.related_event_ids) == len(set(issue.related_event_ids))
    assert issue.description


def test_related_ids_are_in_trace_order():
    events = _failed_attempts(_OP_A, 3)
    issue = RetryDetector().detect(_RUN, events)[0]
    assert issue.related_event_ids == [e.id for e in events]


def test_metadata_is_json_compatible_and_round_trips():
    issue = _detect(_failed_attempts(_OP_A, 3))[0]
    assert issue.metadata["detector"] == "retry_detector"
    assert issue.metadata["operation"] == "fetch_customer"
    assert issue.metadata["failed_attempts"] == 3
    assert issue.metadata["input"] == {"customer_id": "123"}

    dumped = json.dumps(issue.metadata)
    assert json.loads(dumped) == issue.metadata

    restored = AgentIssue.model_validate_json(issue.model_dump_json())
    assert restored == issue

"""Deterministic detector for unnecessary or wasteful agent work.

Definition
----------
This detector flags a **successful tool operation** only when the trace carries
*explicit structured evidence* that the operation was unnecessary, broader than
required, or wasted. It never infers inefficiency from timing, token counts,
durations, record counts, decision names, LLM calls, or semantics -- only from
boolean / equality signals in event ``metadata`` and mapping ``output``.

Evidence signals (each counted at most once per operation)
--------------------------------------------------------
* ``scope_mismatch`` -- the operation's own ``TOOL_CALL_STARTED`` or
  ``TOOL_CALL_COMPLETED`` has a mapping with ``required_scope`` and
  ``actual_scope`` both present, both non-null, of the same JSON type, and
  unequal. No ordering between scope values is assumed -- only the explicit
  mismatch matters.
* ``explicit_wasted_work`` -- that operation's own start/completed event has a
  mapping with ``wasted_work is True``.
* ``explicit_unnecessary`` -- that operation's own start/completed event has a
  mapping with ``necessary is False``.
* ``explicit_wasted_operation`` -- a later event names this operation via
  ``wasted_operation == <op name>`` (a string). The recognition is associated
  with the most recent preceding successful operation of that name.
* ``sufficient_information_already_available`` -- an event *before* this operation
  started has ``sufficient_to_answer is True`` in a mapping.

Only actual booleans count: ``"true"``, ``1``, ``"false"`` etc. are ignored.

An operation is flagged only if it has at least one of the first four
(*triggering*) signals. ``sufficient_information_already_available`` alone is
context, not an accusation, so it never triggers an issue on its own -- it only
raises severity of an already-flagged operation.

Severity (by number of distinct signals present)
------------------------------------------------
* 1 signal  -> ``MEDIUM``
* 2 signals -> ``HIGH``
* 3+ signals -> ``CRITICAL``

Algorithm / purity
------------------
Sort a copy of the events by ``sequence_number`` (the caller's list is never
reordered; timestamps are never used), pair successful tool calls, then gather
evidence per operation with bounded look-behind / look-ahead scans. Overall
O(n^2) worst case, O(n) in practice for the small traces AgentLens handles.

The detector holds no state between calls and mutates neither the run, the
events, nor any nested ``input`` / ``output`` / ``metadata`` structure.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass

from agentlens.detectors.utils import canonical_json
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

__all__ = ["InefficiencyDetector"]

_LIFECYCLE_EVENT_TYPES = frozenset({EventType.RUN_STARTED, EventType.RUN_COMPLETED})

_SCOPE_MISMATCH = "scope_mismatch"
_WASTED_WORK = "explicit_wasted_work"
_UNNECESSARY = "explicit_unnecessary"
_WASTED_OPERATION = "explicit_wasted_operation"
_SUFFICIENT_INFO = "sufficient_information_already_available"

#: Fixed reporting order for the ``evidence`` metadata list.
_EVIDENCE_ORDER = (
    _SCOPE_MISMATCH,
    _WASTED_WORK,
    _UNNECESSARY,
    _WASTED_OPERATION,
    _SUFFICIENT_INFO,
)
#: Signals that on their own justify emitting an issue.
_TRIGGERING_EVIDENCE = frozenset({_SCOPE_MISMATCH, _WASTED_WORK, _UNNECESSARY, _WASTED_OPERATION})


@dataclass
class _Operation:
    """One successful ``TOOL_CALL_STARTED`` -> ``TOOL_CALL_COMPLETED`` pair."""

    name: str
    started: AgentEvent
    completed: AgentEvent
    index: int  # position among successful operations, in trace order


class InefficiencyDetector:
    """Finds tool operations with explicit structural evidence of wasted work.

    Takes no configuration.
    """

    # -- public API ---------------------------------------------------

    def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
        """Return one :class:`AgentIssue` per inefficient operation, in trace order."""

        ordered = sorted(events, key=lambda event: event.sequence_number)
        operations = _successful_operations(ordered)

        issues: list[AgentIssue] = []
        for operation in operations:
            evidence, related = _analyse(operation, ordered, operations)
            if evidence & _TRIGGERING_EVIDENCE:
                issues.append(_build_issue(run, operation, evidence, related))
        return issues


# ---------------------------------------------------------------------------
# Tool-call pairing
# ---------------------------------------------------------------------------


def _successful_operations(ordered: list[AgentEvent]) -> list[_Operation]:
    operations: list[_Operation] = []
    active: AgentEvent | None = None
    for event in ordered:
        event_type = event.event_type
        if event_type in _LIFECYCLE_EVENT_TYPES:
            continue
        if event_type is EventType.TOOL_CALL_STARTED:
            active = event  # a previous unresolved start is discarded
        elif event_type is EventType.TOOL_CALL_COMPLETED:
            if active is not None and _pair_is_valid(active, event):
                operations.append(_Operation(active.name, active, event, len(operations)))
            active = None
        else:
            # DECISION / ERROR / LLM_* between START and COMPLETED breaks the pair
            active = None
    return operations


def _pair_is_valid(started: AgentEvent, completed: AgentEvent) -> bool:
    if completed.name != started.name:
        return False
    if completed.input is None:
        return True
    return canonical_json(completed.input) == canonical_json(started.input)


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------


def _analyse(
    operation: _Operation,
    ordered: list[AgentEvent],
    operations: list[_Operation],
) -> tuple[set[str], list[AgentEvent]]:
    evidence: set[str] = set()
    related: list[AgentEvent] = [operation.started, operation.completed]

    for event in (operation.started, operation.completed):
        for mapping in _mappings(event):
            if _scope_mismatch(mapping):
                evidence.add(_SCOPE_MISMATCH)
            if mapping.get("wasted_work") is True:
                evidence.add(_WASTED_WORK)
            if mapping.get("necessary") is False:
                evidence.add(_UNNECESSARY)

    recognition = _find_wasted_operation_recognition(operation, ordered, operations)
    if recognition is not None:
        evidence.add(_WASTED_OPERATION)
        related.append(recognition)

    sufficiency = _find_prior_sufficiency(operation, ordered)
    if sufficiency is not None:
        evidence.add(_SUFFICIENT_INFO)
        related.append(sufficiency)

    return evidence, related


def _mappings(event: AgentEvent) -> Iterator[dict]:
    """Yield the JSON *object* signal sources of an event: metadata, then output."""

    yield event.metadata
    if isinstance(event.output, dict):
        yield event.output


def _scope_mismatch(mapping: dict) -> bool:
    if "required_scope" not in mapping or "actual_scope" not in mapping:
        return False
    required = mapping["required_scope"]
    actual = mapping["actual_scope"]
    if required is None or actual is None:
        return False
    if type(required) is not type(actual):
        return False
    return required != actual


def _find_wasted_operation_recognition(
    operation: _Operation,
    ordered: list[AgentEvent],
    operations: list[_Operation],
) -> AgentEvent | None:
    """Earliest post-completion event naming this operation as wasted, unless a
    later successful operation of the same name claims that recognition first.
    """

    later_same_name_completions = [
        other.completed.sequence_number
        for other in operations
        if other.index > operation.index and other.name == operation.name
    ]
    for event in ordered:
        if event.sequence_number <= operation.completed.sequence_number:
            continue
        if not _names_wasted_operation(event, operation.name):
            continue
        claimed_by_later_operation = any(
            completed_seq < event.sequence_number for completed_seq in later_same_name_completions
        )
        if not claimed_by_later_operation:
            return event
    return None


def _names_wasted_operation(event: AgentEvent, name: str) -> bool:
    for mapping in _mappings(event):
        value = mapping.get("wasted_operation")
        if isinstance(value, str) and value == name:
            return True
    return False


def _find_prior_sufficiency(operation: _Operation, ordered: list[AgentEvent]) -> AgentEvent | None:
    """Most recent event before this operation started that explicitly marked the
    available information as sufficient.
    """

    found: AgentEvent | None = None
    for event in ordered:
        if event.sequence_number >= operation.started.sequence_number:
            break
        for mapping in _mappings(event):
            if mapping.get("sufficient_to_answer") is True:
                found = event
    return found


# ---------------------------------------------------------------------------
# Issue construction
# ---------------------------------------------------------------------------


def _build_issue(
    run: AgentRun,
    operation: _Operation,
    evidence: set[str],
    related: list[AgentEvent],
) -> AgentIssue:
    ordered_evidence = [tag for tag in _EVIDENCE_ORDER if tag in evidence]
    return AgentIssue(
        run_id=run.id,
        issue_type=IssueType.INEFFICIENCY,
        severity=_severity_for(len(ordered_evidence)),
        description=(
            f"Detected unnecessary or excessive work in tool operation {operation.name!r}."
        ),
        related_event_ids=_ordered_unique_ids(related),
        metadata={
            "detector": "inefficiency_detector",
            "operation": operation.name,
            "input": deepcopy(operation.started.input),
            "evidence": ordered_evidence,
        },
    )


def _ordered_unique_ids(events: list[AgentEvent]) -> list:
    seen: set = set()
    result = []
    for event in sorted(events, key=lambda e: e.sequence_number):
        if event.id not in seen:
            seen.add(event.id)
            result.append(event.id)
    return result


def _severity_for(signal_count: int) -> Severity:
    if signal_count >= 3:
        return Severity.CRITICAL
    if signal_count == 2:
        return Severity.HIGH
    return Severity.MEDIUM

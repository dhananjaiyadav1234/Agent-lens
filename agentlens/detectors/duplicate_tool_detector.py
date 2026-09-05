"""Deterministic detector for redundant repeated tool calls.

Definition
----------
A *duplicate tool call* is a **second (or later) successful** execution of the
same logical tool operation -- identical tool ``name`` and canonical JSON
``input`` (see :func:`~agentlens.detectors.utils.operation_fingerprint`) -- with
no information-producing event in between that would justify redoing the work.

A *successful* tool call is a ``TOOL_CALL_STARTED`` immediately followed, in
``sequence_number`` order, by a ``TOOL_CALL_COMPLETED`` for a consistent
operation. Anything between them (a ``DECISION``, an ``ERROR``, another
``TOOL_CALL_STARTED``) invalidates the pair. Failed attempts and retries belong
to :class:`~agentlens.detectors.retry_detector.RetryDetector`, not here.

Information boundaries
---------------------
The detector keeps a set of operations already completed successfully in the
current *information generation*. The generation is reset (that set is cleared)
by, and only by:

* a successful ``TOOL_CALL_COMPLETED`` for a **different** operation -- clears
  every *other* remembered operation (a different tool producing a result is new
  information);
* an ``LLM_CALL_COMPLETED``;
* an event whose ``metadata`` or ``output`` mapping sets one of
  ``uses_new_information`` / ``depends_on_new_information`` /
  ``received_new_information`` / ``new_information`` to boolean ``True``;
* a ``TOOL_CALL_COMPLETED`` that does not form a valid pair (treated
  conservatively as "something happened").

The generation is **not** reset by: a plain ``DECISION``; ``metadata`` /
``output`` that explicitly says ``uses_new_information: false`` (or similar);
lifecycle events; ``ERROR`` events; or the previous successful completion of the
*same* operation.

Algorithm / complexity
----------------------
Sort a copy of the events by ``sequence_number`` (the caller's list is never
reordered; timestamps are never used), then a single left-to-right pass with a
small state machine. O(n log n) for the sort, O(n) for the pass.

The detector is a pure function of its inputs: it mutates neither the run, the
events, nor any nested ``input`` / ``output`` / ``metadata`` structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from agentlens.detectors.utils import canonical_json, operation_fingerprint
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

__all__ = ["DuplicateToolDetector"]

_LIFECYCLE_EVENT_TYPES = frozenset({EventType.RUN_STARTED, EventType.RUN_COMPLETED})

# Keys (in an event's metadata or output mapping) that, when set to boolean True,
# mark the event as an explicit information boundary.
_NEW_INFORMATION_KEYS = (
    "uses_new_information",
    "depends_on_new_information",
    "received_new_information",
    "new_information",
)

_Pair = tuple[AgentEvent, AgentEvent]


class DuplicateToolDetector:
    """Finds redundant repeated successful tool calls within a run.

    Takes no configuration: a duplicate is structurally defined as the second
    successful call of an identical operation with no intervening new
    information.
    """

    # -- public API ---------------------------------------------------

    def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
        """Return one :class:`AgentIssue` per redundant duplicate call, in trace order."""

        ordered = sorted(events, key=lambda event: event.sequence_number)

        issues: list[AgentIssue] = []
        active: AgentEvent | None = None
        first_success: dict[str, _Pair] = {}
        duplicate_counts: dict[str, int] = {}

        def advance_generation() -> None:
            first_success.clear()
            duplicate_counts.clear()

        for event in ordered:
            event_type = event.event_type

            if event_type in _LIFECYCLE_EVENT_TYPES:
                continue

            if event_type is EventType.TOOL_CALL_STARTED:
                active = event  # a previous unresolved start is discarded
                continue

            if event_type is EventType.TOOL_CALL_COMPLETED:
                pair: _Pair | None = None
                if active is not None and _pair_is_valid(active, event):
                    pair = (active, event)
                active = None

                if pair is None:
                    advance_generation()  # ambiguous completion -> conservative boundary
                    continue

                fingerprint = operation_fingerprint(pair[0].name, pair[0].input)
                if fingerprint in first_success:
                    duplicate_counts[fingerprint] += 1
                    issues.append(
                        self._build_issue(
                            run=run,
                            original=first_success[fingerprint],
                            duplicate=pair,
                            duplicate_count=duplicate_counts[fingerprint],
                        )
                    )
                else:
                    first_success[fingerprint] = pair
                    duplicate_counts[fingerprint] = 0

                # A successful result for this operation is new information for
                # every other operation: forget them.
                for other in [key for key in first_success if key != fingerprint]:
                    del first_success[other]
                    del duplicate_counts[other]
                continue

            # DECISION, ERROR, LLM_CALL_*, or anything else: any pending start is
            # abandoned; some of these advance the information generation.
            active = None
            if event_type is EventType.LLM_CALL_COMPLETED or _indicates_new_information(event):
                advance_generation()

        return issues

    # -- internals --------------------------------------------------

    def _build_issue(
        self,
        *,
        run: AgentRun,
        original: _Pair,
        duplicate: _Pair,
        duplicate_count: int,
    ) -> AgentIssue:
        original_started, original_completed = original
        duplicate_started, duplicate_completed = duplicate
        name = original_started.name
        return AgentIssue(
            run_id=run.id,
            issue_type=IssueType.DUPLICATE,
            severity=_severity_for(duplicate_count),
            description=(
                f"Detected duplicate successful tool call to {name!r} with identical input."
            ),
            related_event_ids=[
                original_started.id,
                original_completed.id,
                duplicate_started.id,
                duplicate_completed.id,
            ],
            metadata={
                "detector": "duplicate_tool_detector",
                "operation": name,
                "input": deepcopy(original_started.input),
                "original_sequence_number": original_started.sequence_number,
                "duplicate_sequence_number": duplicate_started.sequence_number,
                "duplicate_count": duplicate_count,
            },
        )


def _pair_is_valid(started: AgentEvent, completed: AgentEvent) -> bool:
    """Whether ``started`` and ``completed`` describe one successful operation.

    Conservative: the names must match, and if ``completed`` carries its own
    ``input`` it must be canonically equal to ``started.input``.
    """

    if completed.name != started.name:
        return False
    if completed.input is None:
        return True
    return canonical_json(completed.input) == canonical_json(started.input)


def _indicates_new_information(event: AgentEvent) -> bool:
    """Whether ``event`` explicitly, deterministically signals new information."""

    for source in (event.metadata, event.output):
        if isinstance(source, dict):
            for key in _NEW_INFORMATION_KEYS:
                if source.get(key) is True:
                    return True
    return False


def _severity_for(duplicate_count: int) -> Severity:
    """Deterministic duplicate-ordinal -> severity mapping.

    * 1st duplicate       -> MEDIUM
    * 2nd or 3rd duplicate -> HIGH
    * 4th or later         -> CRITICAL
    """

    if duplicate_count >= 4:
        return Severity.CRITICAL
    if duplicate_count >= 2:
        return Severity.HIGH
    return Severity.MEDIUM

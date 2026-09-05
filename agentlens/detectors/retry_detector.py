"""Deterministic detector for excessive retries of the same tool operation.

Definition
----------
A *failed attempt* is a ``TOOL_CALL_STARTED`` immediately followed (in
``sequence_number`` order) by an ``ERROR`` that is associated with it. A *retry
streak* is two or more consecutive failed attempts of the **same logical
operation** -- identified by tool ``name`` plus canonical JSON of ``input`` (see
:func:`~agentlens.detectors.utils.operation_fingerprint`). When a streak reaches
``minimum_failed_attempts`` failed attempts it produces one
:class:`~agentlens.models.AgentIssue` describing those failures.

The eventual successful attempt (``TOOL_CALL_STARTED`` -> ``TOOL_CALL_COMPLETED``)
ends the streak but is **not** part of it: it is neither counted nor referenced.

Algorithm
---------
Events are analysed in ``sequence_number`` order (a sorted copy -- the caller's
list is never reordered). A small state machine tracks:

* ``active`` -- a ``TOOL_CALL_STARTED`` awaiting its outcome;
* ``streak`` -- consecutive failed attempts accumulated for one operation.

For each event:

* ``TOOL_CALL_STARTED`` -- if a previous attempt is still unresolved, or the
  streak is for a different operation, flush the streak first; then open a new
  active attempt.
* ``ERROR`` -- only meaningful when it directly follows an active attempt. If it
  is associated with that attempt (see below) it is a failed attempt and extends
  the streak; a mismatched ``ERROR.operation`` flushes the streak instead. An
  ``ERROR`` with no active attempt flushes the streak.
* ``TOOL_CALL_COMPLETED`` for the active operation -- a success: flush the streak
  (reporting the prior failures) and clear the active attempt.
* anything else -- ``DECISION``, ``LLM_CALL_*``, ``RUN_STARTED`` / ``RUN_COMPLETED``
  lifecycle boundaries, an unrelated ``TOOL_CALL_COMPLETED`` -- flushes the streak
  and clears any active attempt. Retry streaks never span these.

"Flush" emits an issue iff the streak has reached the threshold, then resets it,
so a region is never reported twice.

Complexity is O(n log n) for the sort plus O(n) for the single pass. The
detector mutates neither the run, the events, nor any nested structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from agentlens.detectors.utils import operation_fingerprint
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity
from agentlens.models.base import JsonValue

__all__ = ["RetryDetector"]

_DEFAULT_MINIMUM_FAILED_ATTEMPTS = 3


class _ActiveAttempt:
    """A ``TOOL_CALL_STARTED`` whose outcome has not yet been seen."""

    __slots__ = ("event", "fingerprint", "name", "input")

    def __init__(self, event: AgentEvent, fingerprint: str) -> None:
        self.event = event
        self.fingerprint = fingerprint
        self.name = event.name
        self.input = event.input


class _RetryStreak:
    """Consecutive failed attempts accumulated for one operation."""

    __slots__ = ("fingerprint", "name", "input", "events", "failed_attempts")

    def __init__(self) -> None:
        self.fingerprint: str | None = None
        self.name: str | None = None
        self.input: JsonValue | None = None
        self.events: list[AgentEvent] = []
        self.failed_attempts = 0

    def begin(self, attempt: _ActiveAttempt) -> None:
        self.fingerprint = attempt.fingerprint
        self.name = attempt.name
        self.input = attempt.input

    def add_failure(self, started: AgentEvent, error: AgentEvent) -> None:
        self.events.append(started)
        self.events.append(error)
        self.failed_attempts += 1


class RetryDetector:
    """Finds runs of consecutive failed attempts of the same tool operation.

    Parameters
    ----------
    minimum_failed_attempts:
        How many consecutive failed attempts of one operation count as an
        excessive retry. Must be an ``int`` >= 2. Defaults to 3. Invalid values
        raise immediately; nothing is coerced.
    """

    def __init__(self, minimum_failed_attempts: int = _DEFAULT_MINIMUM_FAILED_ATTEMPTS) -> None:
        if isinstance(minimum_failed_attempts, bool) or not isinstance(
            minimum_failed_attempts, int
        ):
            raise TypeError(
                "minimum_failed_attempts must be an int, got "
                f"{type(minimum_failed_attempts).__name__!r}"
            )
        if minimum_failed_attempts < 2:
            raise ValueError(
                f"minimum_failed_attempts must be at least 2, got {minimum_failed_attempts}"
            )
        self.minimum_failed_attempts = minimum_failed_attempts

    # -- public API ---------------------------------------------------

    def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
        """Return one :class:`AgentIssue` per excessive retry streak, in trace order."""

        ordered = sorted(events, key=lambda event: event.sequence_number)

        issues: list[AgentIssue] = []
        active: _ActiveAttempt | None = None
        streak = _RetryStreak()

        def flush() -> None:
            nonlocal streak
            if streak.failed_attempts >= self.minimum_failed_attempts:
                issues.append(self._build_issue(run, streak))
            streak = _RetryStreak()

        for event in ordered:
            event_type = event.event_type

            if event_type is EventType.TOOL_CALL_STARTED:
                fingerprint = operation_fingerprint(event.name, event.input)
                # An unresolved previous attempt, or a streak for a different
                # operation, both end the current streak before this one opens.
                unresolved_previous = active is not None
                switches_operation = (
                    streak.fingerprint is not None and streak.fingerprint != fingerprint
                )
                if unresolved_previous or switches_operation:
                    flush()
                active = _ActiveAttempt(event, fingerprint)
                continue

            if event_type is EventType.ERROR:
                if active is None:
                    flush()
                    continue
                if _error_is_associated(event, active.name):
                    if streak.fingerprint is not None and streak.fingerprint != active.fingerprint:
                        flush()
                    if streak.fingerprint is None:
                        streak.begin(active)
                    streak.add_failure(active.event, event)
                else:
                    flush()
                active = None
                continue

            # TOOL_CALL_COMPLETED (a success, for this operation or another),
            # DECISION, LLM_CALL_*, RUN_STARTED / RUN_COMPLETED lifecycle
            # boundaries, or anything else: all end the current streak. A success
            # for the active operation is the common "retried, then it worked"
            # case -- the prior failures are still reported by flush().
            flush()
            active = None

        flush()
        return issues

    # -- internals --------------------------------------------------

    def _build_issue(self, run: AgentRun, streak: _RetryStreak) -> AgentIssue:
        failed = streak.failed_attempts
        return AgentIssue(
            run_id=run.id,
            issue_type=IssueType.RETRY,
            severity=_severity_for(failed),
            description=(
                f"Detected {failed} consecutive failed attempts of tool operation {streak.name!r}."
            ),
            related_event_ids=[event.id for event in streak.events],
            metadata={
                "detector": "retry_detector",
                "operation": streak.name,
                "failed_attempts": failed,
                "input": deepcopy(streak.input),
            },
        )


def _error_is_associated(error_event: AgentEvent, active_name: str) -> bool:
    """Decide whether ``error_event`` belongs to the active tool attempt.

    * ``output`` is a mapping with a string ``"operation"`` -> associated only if
      that value equals ``active_name``.
    * ``output`` is a mapping with a non-string ``"operation"`` -> not associated
      (conservative).
    * ``output`` has no ``"operation"`` (or is not a mapping at all) -> associated,
      because the ERROR directly follows the active attempt.
    """

    output = error_event.output
    if isinstance(output, dict) and "operation" in output:
        operation = output["operation"]
        if isinstance(operation, str):
            return operation == active_name
        return False
    return True


def _severity_for(failed_attempts: int) -> Severity:
    """Deterministic failed-attempt-count -> severity mapping.

    * 2 or 3 failed attempts -> MEDIUM
    * 4 or 5 failed attempts -> HIGH
    * 6 or more              -> CRITICAL
    """

    if failed_attempts >= 6:
        return Severity.CRITICAL
    if failed_attempts >= 4:
        return Severity.HIGH
    return Severity.MEDIUM

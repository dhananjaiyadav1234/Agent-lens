"""Deterministic detector for pathological repeated execution cycles.

Definition
----------
A *loop* is a contiguous run of one or more logically-equivalent event
fingerprints (see :func:`~agentlens.detectors.utils.event_fingerprint`) repeated
consecutively at least ``minimum_repetitions`` times. ``A B A B A B`` is a loop
with ``cycle_length=2`` and ``repetitions=3``; ``A A A`` is a loop with
``cycle_length=1`` and ``repetitions=3``.

Algorithm
---------
1. Copy the caller's events and sort the copy by ``sequence_number`` (the
   authoritative ordering key). The caller's list is never reordered.
2. Split the sorted events into *segments*:

   * ``RUN_STARTED`` / ``RUN_COMPLETED`` lifecycle events are dropped entirely --
     they never participate in a cycle;
   * ``ERROR`` events act as hard segment boundaries and are themselves dropped.
     Repeated failure/retry sequences (``TOOL_CALL_STARTED, ERROR, ...``) are a
     *retry* pattern, not a loop; a dedicated ``RetryDetector`` will own them in a
     later milestone. Breaking segments on ``ERROR`` keeps this detector from
     claiming them.
3. Within each segment, scan left to right. At each position try the *shortest*
   cycle length first; count how many times that cycle repeats consecutively; if
   it repeats at least ``minimum_repetitions`` times, emit one issue for the
   whole repeated region and advance past it. Otherwise advance one event.

   Preferring the shortest cycle and then skipping the region means ``A A A A A A``
   yields a single ``cycle_length=1, repetitions=6`` issue rather than several
   overlapping ones.

Complexity is roughly O(n^3 / minimum_repetitions) per segment in the worst
case, which is fine for the modest trace sizes AgentLens handles today.

The detector is a pure function of its inputs: it mutates neither the run, the
events, nor any nested ``input`` / ``output`` / ``metadata`` structure.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentlens.detectors.utils import event_fingerprint
from agentlens.models import AgentEvent, AgentIssue, AgentRun, EventType, IssueType, Severity

__all__ = ["LoopDetector"]

_DEFAULT_MINIMUM_REPETITIONS = 3

# Lifecycle events are never part of a cycle.
_LIFECYCLE_EVENT_TYPES = frozenset({EventType.RUN_STARTED, EventType.RUN_COMPLETED})


class LoopDetector:
    """Finds repeated contiguous execution cycles in a trace.

    Parameters
    ----------
    minimum_repetitions:
        How many consecutive repetitions of a cycle count as a loop. Must be an
        ``int`` >= 2. Defaults to 3. Invalid values raise immediately; nothing is
        coerced.
    """

    def __init__(self, minimum_repetitions: int = _DEFAULT_MINIMUM_REPETITIONS) -> None:
        if isinstance(minimum_repetitions, bool) or not isinstance(minimum_repetitions, int):
            raise TypeError(
                f"minimum_repetitions must be an int, got {type(minimum_repetitions).__name__!r}"
            )
        if minimum_repetitions < 2:
            raise ValueError(f"minimum_repetitions must be at least 2, got {minimum_repetitions}")
        self.minimum_repetitions = minimum_repetitions

    # -- public API ---------------------------------------------------

    def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
        """Return one :class:`AgentIssue` per distinct detected loop, in trace order."""

        ordered = sorted(events, key=lambda event: event.sequence_number)

        issues: list[AgentIssue] = []
        segment: list[AgentEvent] = []
        for event in ordered:
            if event.event_type in _LIFECYCLE_EVENT_TYPES:
                continue
            if event.event_type is EventType.ERROR:
                issues.extend(self._detect_in_segment(run, segment))
                segment = []
                continue
            segment.append(event)
        issues.extend(self._detect_in_segment(run, segment))
        return issues

    # -- internals --------------------------------------------------

    def _detect_in_segment(self, run: AgentRun, segment: list[AgentEvent]) -> list[AgentIssue]:
        fingerprints = [event_fingerprint(event) for event in segment]
        n = len(fingerprints)

        issues: list[AgentIssue] = []
        start = 0
        while start < n:
            emitted = False
            max_cycle_length = (n - start) // self.minimum_repetitions
            for cycle_length in range(1, max_cycle_length + 1):
                repetitions = _count_repetitions(fingerprints, start, cycle_length)
                if repetitions >= self.minimum_repetitions:
                    span = cycle_length * repetitions
                    issues.append(
                        self._build_issue(
                            run=run,
                            cycle=segment[start : start + cycle_length],
                            region=segment[start : start + span],
                            cycle_length=cycle_length,
                            repetitions=repetitions,
                        )
                    )
                    start += span
                    emitted = True
                    break
            if not emitted:
                start += 1
        return issues

    def _build_issue(
        self,
        *,
        run: AgentRun,
        cycle: list[AgentEvent],
        region: list[AgentEvent],
        cycle_length: int,
        repetitions: int,
    ) -> AgentIssue:
        pattern = [{"event_type": event.event_type.value, "name": event.name} for event in cycle]
        plural = "s" if cycle_length != 1 else ""
        return AgentIssue(
            run_id=run.id,
            issue_type=IssueType.LOOP,
            severity=_severity_for(repetitions),
            description=(
                f"Detected a repeated execution cycle of {cycle_length} event{plural} "
                f"repeated {repetitions} times."
            ),
            related_event_ids=[event.id for event in region],
            metadata={
                "detector": "loop_detector",
                "cycle_length": cycle_length,
                "repetitions": repetitions,
                "pattern": pattern,
            },
        )


def _count_repetitions(fingerprints: list[str], start: int, cycle_length: int) -> int:
    """Count how many times ``fingerprints[start:start+cycle_length]`` repeats
    consecutively starting at ``start``.
    """

    cycle = fingerprints[start : start + cycle_length]
    repetitions = 0
    pos = start
    while fingerprints[pos : pos + cycle_length] == cycle:
        repetitions += 1
        pos += cycle_length
    return repetitions


def _severity_for(repetitions: int) -> Severity:
    """Deterministic repetition-count -> severity mapping.

    * 3 repetitions      -> MEDIUM
    * 4 or 5 repetitions -> HIGH
    * 6 or more          -> CRITICAL
    """

    if repetitions >= 6:
        return Severity.CRITICAL
    if repetitions >= 4:
        return Severity.HIGH
    return Severity.MEDIUM

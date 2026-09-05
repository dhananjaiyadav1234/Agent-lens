"""Run every deterministic detector over one trace, in a fixed order.

This is the whole of "orchestration": a fixed tuple of detector classes and a
function that runs each with its default configuration and concatenates the
results. No registry, no plugins, no configuration -- four deterministic
detectors do not need more than this.

``run_detectors`` is framework-agnostic like the detectors themselves: it works
on any ``AgentRun`` + event sequence, whether produced by AgentLens or imported
from elsewhere. :meth:`agentlens.AgentLens.detect` is the convenience wrapper
that looks the run and its events up in a live tracing store first.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentlens.detectors.duplicate_tool_detector import DuplicateToolDetector
from agentlens.detectors.inefficiency_detector import InefficiencyDetector
from agentlens.detectors.loop_detector import LoopDetector
from agentlens.detectors.retry_detector import RetryDetector
from agentlens.models import AgentEvent, AgentIssue, AgentRun

__all__ = ["DEFAULT_DETECTOR_CLASSES", "run_detectors"]

#: The deterministic detectors, in the fixed order their issues are returned.
DEFAULT_DETECTOR_CLASSES: tuple[type, ...] = (
    LoopDetector,
    RetryDetector,
    DuplicateToolDetector,
    InefficiencyDetector,
)


def run_detectors(run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
    """Run every default detector over ``run`` / ``events`` and concatenate results.

    Each detector in :data:`DEFAULT_DETECTOR_CLASSES` is instantiated with its
    default configuration and given the *same* ``run`` and ``events``. The result
    is the concatenation of each detector's issues, in detector order; within a
    detector the issues keep that detector's own (trace) ordering.

    Equivalent to::

        [
            *LoopDetector().detect(run, events),
            *RetryDetector().detect(run, events),
            *DuplicateToolDetector().detect(run, events),
            *InefficiencyDetector().detect(run, events),
        ]

    Pure: it mutates neither ``run`` nor ``events``, keeps no state between calls,
    and performs no I/O.
    """

    issues: list[AgentIssue] = []
    for detector_cls in DEFAULT_DETECTOR_CLASSES:
        issues.extend(detector_cls().detect(run, events))
    return issues

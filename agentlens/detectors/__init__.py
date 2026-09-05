"""AgentLens deterministic issue detectors.

Detectors are pure analysis components. Each takes an
:class:`~agentlens.models.AgentRun` plus its ordered
:class:`~agentlens.models.AgentEvent` list and returns validated
:class:`~agentlens.models.AgentIssue` objects. They depend only on the universal
models -- not on ``AgentLens``, storage, or any framework -- and perform no I/O.

Usage::

    from agentlens.detectors import run_detectors

    issues = run_detectors(run, events)  # all four detectors, fixed order

    # or run one directly:
    from agentlens.detectors import LoopDetector

    issues = LoopDetector().detect(run, events)
"""

from __future__ import annotations

from agentlens.detectors.base import IssueDetector
from agentlens.detectors.duplicate_tool_detector import DuplicateToolDetector
from agentlens.detectors.inefficiency_detector import InefficiencyDetector
from agentlens.detectors.loop_detector import LoopDetector
from agentlens.detectors.orchestration import DEFAULT_DETECTOR_CLASSES, run_detectors
from agentlens.detectors.retry_detector import RetryDetector

__all__ = [
    "IssueDetector",
    "LoopDetector",
    "RetryDetector",
    "DuplicateToolDetector",
    "InefficiencyDetector",
    "run_detectors",
    "DEFAULT_DETECTOR_CLASSES",
]

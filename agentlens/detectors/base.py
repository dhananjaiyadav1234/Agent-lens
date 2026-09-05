"""The framework-independent detector interface.

A detector is a pure analysis component: it takes a run plus its ordered events
and returns validated :class:`~agentlens.models.AgentIssue` objects. It depends
only on the universal models -- never on ``AgentLens``, storage, the tracing
context, or any agent framework -- so the same detector works on AgentLens
traces, imported JSON traces, or future framework adapters alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentlens.models import AgentEvent, AgentIssue, AgentRun

__all__ = ["IssueDetector"]


@runtime_checkable
class IssueDetector(Protocol):
    """Structural type for a deterministic issue detector.

    Implementations must:

    * depend only on the universal models (no storage, no framework);
    * not mutate ``run``, ``events``, or anything reachable from them;
    * perform no network or file I/O;
    * be deterministic -- the same inputs always yield the same issues;
    * return validated ``AgentIssue`` instances (possibly an empty list).
    """

    def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
        """Analyze ``run`` and its ``events`` and return any issues found."""
        ...

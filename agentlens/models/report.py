"""Read-only aggregate report over a stored run, its events, and its issues.

Nothing in this module runs detectors, writes storage, generates ids, or reads
the clock. :class:`AgentReport` and :class:`ReportSummary` are plain validated
values derived entirely from data already retrieved from a
:class:`~agentlens.core.storage.TraceStore`.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

from pydantic import Field

from agentlens.models.base import AgentLensModel
from agentlens.models.enums import EventType, IssueType, Severity
from agentlens.models.event import AgentEvent
from agentlens.models.issue import AgentIssue
from agentlens.models.run import AgentRun

__all__ = ["ReportSummary", "AgentReport"]


def _counts_in_first_appearance_order(keys: Iterable[Hashable]) -> dict:
    """Count ``keys``, keeping each key in the order it is first seen."""

    counts: dict = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


class ReportSummary(AgentLensModel):
    """Deterministic aggregate counts for one run's events and persisted issues.

    Every field is a pure function of the event and issue lists -- no timestamps,
    no ids, no ranking. The ``*_by_*`` mappings contain only the keys actually
    present, ordered by first appearance in the source list (never sorted).
    """

    total_events: int = Field(
        description="Number of events in the run; lifecycle events are included (``len(events)``)."
    )
    total_issues: int = Field(
        description="Number of persisted issues for the run; no de-duplication (``len(issues)``)."
    )
    events_by_type: dict[EventType, int] = Field(
        default_factory=dict,
        description="Event count per EventType, keyed in first-appearance order.",
    )
    issues_by_type: dict[IssueType, int] = Field(
        default_factory=dict,
        description="Issue count per IssueType, keyed in first-appearance order.",
    )
    issues_by_severity: dict[Severity, int] = Field(
        default_factory=dict,
        description="Issue count per Severity, keyed in first-appearance order.",
    )

    @classmethod
    def from_events_and_issues(
        cls,
        events: Sequence[AgentEvent],
        issues: Sequence[AgentIssue],
    ) -> ReportSummary:
        """Build a summary from the run's events and its persisted issues (in order)."""

        return cls(
            total_events=len(events),
            total_issues=len(issues),
            events_by_type=_counts_in_first_appearance_order(e.event_type for e in events),
            issues_by_type=_counts_in_first_appearance_order(i.issue_type for i in issues),
            issues_by_severity=_counts_in_first_appearance_order(i.severity for i in issues),
        )


class AgentReport(AgentLensModel):
    """An aggregated, read-only view of one stored run.

    Built by :meth:`agentlens.AgentLens.get_report`. It carries the run, its
    events and its persisted issues verbatim -- in exactly the order the store
    returned them -- plus a deterministic :class:`ReportSummary`.
    """

    run: AgentRun
    events: list[AgentEvent] = Field(default_factory=list)
    issues: list[AgentIssue] = Field(default_factory=list)
    summary: ReportSummary

    @classmethod
    def build(
        cls,
        run: AgentRun,
        events: Sequence[AgentEvent],
        issues: Sequence[AgentIssue],
    ) -> AgentReport:
        """Assemble a validated report; ``events`` / ``issues`` order is preserved."""

        return cls(
            run=run,
            events=list(events),
            issues=list(issues),
            summary=ReportSummary.from_events_and_issues(events, issues),
        )

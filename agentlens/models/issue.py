"""The :class:`AgentIssue` model: a problem detected during a run."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import Field

from agentlens.models.base import AgentLensModel, JsonMapping
from agentlens.models.enums import IssueType, Severity

__all__ = ["AgentIssue"]


class AgentIssue(AgentLensModel):
    """A single problem attributed to a run by a detector or by LLM analysis.

    Issues are additive findings, not part of the run's own lifecycle: producing
    an issue never mutates the :class:`~agentlens.models.AgentRun` or its events.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this issue.")
    run_id: UUID = Field(description="Identifier of the AgentRun this issue was found in.")
    issue_type: IssueType = Field(description="Category of the detected problem.")
    severity: Severity = Field(description="How serious the problem is.")
    description: str = Field(description="Human-readable explanation of the problem.")
    related_event_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of the events implicated in this issue (may be empty).",
    )
    metadata: JsonMapping = Field(
        default_factory=dict,
        description="Free-form JSON-compatible context (detector name, scores, ...).",
    )

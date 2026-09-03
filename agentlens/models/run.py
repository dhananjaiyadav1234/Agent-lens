"""The :class:`AgentRun` model: one complete agent execution."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import Field, model_validator

from agentlens.models.base import AgentLensModel, JsonMapping, UtcDatetime, utcnow
from agentlens.models.enums import RunStatus

__all__ = ["AgentRun"]


class AgentRun(AgentLensModel):
    """A single, complete (or in-progress) execution of an agent.

    An ``AgentRun`` is the root of a trace: every :class:`~agentlens.models.AgentEvent`
    and :class:`~agentlens.models.AgentIssue` points back to one via ``run_id``.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this run.")
    task: str = Field(description="What the agent was asked to do.")
    status: RunStatus = Field(
        default=RunStatus.RUNNING,
        description="Lifecycle state of the run.",
    )
    started_at: UtcDatetime = Field(
        default_factory=utcnow,
        description="When the run began (timezone-aware UTC).",
    )
    finished_at: UtcDatetime | None = Field(
        default=None,
        description="When the run ended (timezone-aware UTC); absent while still running.",
    )
    metadata: JsonMapping = Field(
        default_factory=dict,
        description="Free-form JSON-compatible context (model name, tags, cost, ...).",
    )

    @model_validator(mode="after")
    def _check_lifecycle_consistency(self) -> AgentRun:
        """Enforce that ``status`` and ``finished_at`` describe the same lifecycle state.

        * ``RUNNING`` -> ``finished_at`` must be absent (the run has not ended).
        * ``SUCCESS`` / ``FAILED`` -> ``finished_at`` must be present, and must
          not be earlier than ``started_at``.
        """

        if self.status is RunStatus.RUNNING:
            if self.finished_at is not None:
                raise ValueError("finished_at must be None while status is RUNNING")
            return self

        # status is a completed state (SUCCESS or FAILED)
        if self.finished_at is None:
            raise ValueError(f"finished_at is required when status is {self.status.name}")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at")
        return self

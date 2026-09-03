"""The :class:`AgentEvent` model: one event within an :class:`~agentlens.models.AgentRun`."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from agentlens.models.base import AgentLensModel, JsonMapping, JsonValue, UtcDatetime, utcnow
from agentlens.models.enums import EventType

__all__ = ["AgentEvent"]


class AgentEvent(AgentLensModel):
    """A single recorded step in an agent run.

    Events are the atoms deterministic detectors operate on. Ordering within a run
    is defined by ``sequence_number`` (monotonic, gap-tolerant), not by
    ``timestamp`` -- wall-clock time can be coarse or non-monotonic depending on
    the source framework.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this event.")
    run_id: UUID = Field(description="Identifier of the AgentRun this event belongs to.")
    sequence_number: NonNegativeInt = Field(
        description="Position of this event within its run; ordering key (>= 0).",
    )
    timestamp: UtcDatetime = Field(
        default_factory=utcnow,
        description="When the event occurred (timezone-aware UTC).",
    )
    event_type: EventType = Field(description="Which kind of event this is.")
    name: str = Field(
        description="Human-readable label, e.g. the tool or model name.",
    )
    input: JsonValue | None = Field(
        default=None,
        description="JSON-compatible input payload for the step, if any.",
    )
    output: JsonValue | None = Field(
        default=None,
        description="JSON-compatible output payload for the step, if any.",
    )
    duration_ms: NonNegativeFloat | None = Field(
        default=None,
        description="Wall-clock duration of the step in milliseconds (>= 0) when known.",
    )
    status: str | None = Field(
        default=None,
        description=(
            "Optional free-form outcome marker for the step (e.g. 'ok', 'error', "
            "an HTTP status). Left unconstrained so adapters can pass through "
            "framework-native values; not an enum."
        ),
    )
    metadata: JsonMapping = Field(
        default_factory=dict,
        description="Free-form JSON-compatible context for the event.",
    )

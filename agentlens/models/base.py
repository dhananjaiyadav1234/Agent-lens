"""Shared building blocks for the universal AgentLens trace model.

This module deliberately contains no database, HTTP, or agent-framework code. The
models built on top of it form the framework-agnostic contract between future
integrations and the AgentLens core.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, JsonValue

__all__ = ["AgentLensModel", "JsonMapping", "JsonValue", "UtcDatetime", "utcnow"]


# ---------------------------------------------------------------------------
# JSON-compatible data
# ---------------------------------------------------------------------------

# ``JsonValue`` is Pydantic's recursive type for values that are valid JSON:
# ``str``, ``int``, ``float``, ``bool``, ``None``, ``list[JsonValue]`` and
# ``dict[str, JsonValue]``. Fields typed with it reject arbitrary Python objects
# (datetimes, sets, custom classes, ...) at validation time instead of silently
# coercing them. See the module docstring of ``agentlens.models`` for the design
# decision behind this.
JsonMapping = dict[str, JsonValue]
"""A JSON object: string keys mapping to JSON-compatible values."""


# ---------------------------------------------------------------------------
# Timezone-aware UTC datetimes
# ---------------------------------------------------------------------------


def _ensure_utc(value: datetime) -> datetime:
    """Require a timezone-aware datetime and normalise it to UTC.

    Naive datetimes are rejected: without an offset there is no unambiguous
    instant to store. Aware datetimes in other zones are converted to UTC so that
    every timestamp in the model is directly comparable.
    """

    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime must be timezone-aware (received a naive value)")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""A timezone-aware datetime, always stored in UTC."""


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime (used as a default factory)."""

    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class AgentLensModel(BaseModel):
    """Base class for every model in the universal trace contract.

    ``extra="forbid"`` keeps the contract explicit: unrecognised top-level keys
    are an error, and callers that need to attach extra information use the
    dedicated ``metadata`` mapping. ``validate_assignment=True`` means the same
    validation rules apply to attribute mutation after construction.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

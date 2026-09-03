"""The universal AgentLens trace model.

These Pydantic models are the framework-agnostic contract between agent-framework
integrations and the AgentLens core. Everything an adapter captures is normalised
into an :class:`AgentRun` with an ordered stream of :class:`AgentEvent` objects;
detectors and analysis then attach :class:`AgentIssue` findings.

Design decisions
----------------

**JSON-compatible payloads only.** The ``input``, ``output`` and ``metadata``
fields are typed with Pydantic's recursive ``JsonValue`` (str, int, float, bool,
None, list, dict). Arbitrary Python objects are *not* supported at this stage and
are rejected at validation time rather than being silently stringified. Adapters
are responsible for reducing framework objects to plain JSON before constructing
a model. Note that ``float`` permits ``NaN``/``Infinity``, which are not strict
JSON; callers that need strict-JSON guarantees should sanitise upstream.

**Timezones.** Every timestamp is a timezone-aware UTC datetime. Naive datetimes
are rejected; aware datetimes in other zones are converted to UTC on the way in.

**Ordering.** Event order within a run is carried by ``sequence_number``, not by
``timestamp``.

**Strictness.** Models forbid unknown top-level fields and validate on assignment;
use ``metadata`` for anything not covered by an explicit field.
"""

from __future__ import annotations

from agentlens.models.base import AgentLensModel, JsonMapping, JsonValue, UtcDatetime, utcnow
from agentlens.models.enums import EventType, IssueType, RunStatus, Severity
from agentlens.models.event import AgentEvent
from agentlens.models.issue import AgentIssue
from agentlens.models.run import AgentRun

__all__ = [
    # Models
    "AgentRun",
    "AgentEvent",
    "AgentIssue",
    # Enums
    "RunStatus",
    "EventType",
    "IssueType",
    "Severity",
    # Shared helpers / types (useful for adapters and detectors)
    "AgentLensModel",
    "JsonMapping",
    "JsonValue",
    "UtcDatetime",
    "utcnow",
]

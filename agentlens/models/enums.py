"""Enumerations used throughout the universal AgentLens trace model.

All enums derive from :class:`enum.StrEnum` (Python 3.11+) so their members
serialize to plain lowercase strings in JSON and remain human-readable in
storage, logs, and API payloads. Unknown values raise a validation error rather
than being silently accepted.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RunStatus", "EventType", "IssueType", "Severity"]


class RunStatus(StrEnum):
    """Lifecycle state of an :class:`~agentlens.models.AgentRun`."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class EventType(StrEnum):
    """Kind of a single :class:`~agentlens.models.AgentEvent`.

    The set is intentionally small and framework-agnostic. Framework adapters map
    their native event streams onto these members; anything that does not fit is
    recorded on the event ``metadata`` mapping instead of expanding this enum
    prematurely.
    """

    RUN_STARTED = "run_started"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    DECISION = "decision"
    ERROR = "error"
    RUN_COMPLETED = "run_completed"


class IssueType(StrEnum):
    """Category of a problem detected during a run.

    Deterministic detectors and, later, LLM analysis both emit issues tagged with
    one of these types.
    """

    AGENT_LOOP = "agent_loop"
    EXCESSIVE_RETRY = "excessive_retry"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    TOOL_FAILURE = "tool_failure"
    INEFFICIENCY = "inefficiency"


class Severity(StrEnum):
    """How serious a detected issue is, ordered from least to most severe."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

"""AgentLens: framework-agnostic observability, debugging, and deterministic
issue detection for AI agents.

Typical use::

    from agentlens import AgentLens
    from agentlens.models import EventType

    lens = AgentLens()
    with lens.trace("Find customer order") as trace:
        trace.record_event(
            event_type=EventType.DECISION,
            name="select_tool",
            input={"customer_id": "123"},
            output={"tool": "database_lookup"},
        )

Run the deterministic detectors over a completed run with ``lens.detect(run_id)``
(which also persists the issues it generates) and read them back with
``lens.get_issues(run_id)``. Storage is in-memory by default; pass
``AgentLens(store=SQLiteTraceStore(path))`` (see :mod:`agentlens.storage`) to
persist runs, events, and issues across process restarts. Optional framework
adapters (LangChain, the OpenAI Agents SDK, CrewAI) live under
:mod:`agentlens.integrations` and record into an existing run via this same
public API; see the project README for installation and usage.
"""

from __future__ import annotations

from agentlens.core import AgentLens, AgentLensError, RunLifecycleError, TraceStateError

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AgentLens",
    "AgentLensError",
    "TraceStateError",
    "RunLifecycleError",
]

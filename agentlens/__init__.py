"""AgentLens: framework-agnostic observability, debugging, and AI-powered failure
analysis for AI agents.

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
persist runs, events, and issues across process restarts. The REST API and
framework integrations are added in later phases.
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

"""AgentLens core: the framework-agnostic tracing engine.

Public surface:

* :class:`AgentLens` -- the entry point; open traces with ``lens.trace(task)``.
* :class:`Trace` -- the per-run recorder yielded by that context manager.
* :class:`AgentLensError` / :class:`TraceStateError` / :class:`RunLifecycleError`.

``RunManager`` and ``InMemoryTraceStore`` are internal collaborators, exported
here for tests and for the future persistence layer, not for application code.
"""

from __future__ import annotations

from agentlens.core.errors import AgentLensError, RunLifecycleError, TraceStateError
from agentlens.core.lens import AgentLens
from agentlens.core.run_manager import RunManager
from agentlens.core.storage import InMemoryTraceStore, TraceStore
from agentlens.core.tracer import Trace

__all__ = [
    "AgentLens",
    "Trace",
    "RunManager",
    "TraceStore",
    "InMemoryTraceStore",
    "AgentLensError",
    "TraceStateError",
    "RunLifecycleError",
]

"""Persistent trace storage.

:class:`SQLiteTraceStore` is a standard-library-``sqlite3`` implementation of the
:class:`~agentlens.core.storage.TraceStore` contract. Pass one to ``AgentLens``
to keep runs, events, and detected issues across process restarts::

    from agentlens import AgentLens
    from agentlens.storage import SQLiteTraceStore

    lens = AgentLens(store=SQLiteTraceStore("agentlens.db"))

The default (``AgentLens()``) still uses the in-memory store.
"""

from __future__ import annotations

from agentlens.storage.sqlite import SQLiteTraceStore

__all__ = ["SQLiteTraceStore"]

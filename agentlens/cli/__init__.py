"""The ``agentlens`` command-line interface.

A thin, read-only adapter over the existing public API: it opens a
:class:`~agentlens.storage.SQLiteTraceStore`, wraps it in an
:class:`~agentlens.AgentLens`, and calls ``get_run`` / ``get_events`` /
``get_issues`` / ``get_report`` plus ``export_json`` / ``export_markdown``. It
contains no persistence, detection, aggregation, or serialization logic of its
own, and never writes to the database.
"""

from __future__ import annotations

from agentlens.cli.main import build_parser, main

__all__ = ["build_parser", "main"]

"""Read-only export of a built :class:`~agentlens.models.AgentReport`.

Two pure functions -- :func:`export_json` and :func:`export_markdown` -- turn an
``AgentReport`` into a deterministic string. They depend only on
``agentlens.models``: no ``AgentLens``, no storage, no detectors, no I/O. Callers
write the returned string to a file themselves if they want one.
"""

from __future__ import annotations

from agentlens.export.json_export import export_json
from agentlens.export.markdown_export import export_markdown

__all__ = ["export_json", "export_markdown"]

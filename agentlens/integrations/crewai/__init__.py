"""AgentLens ↔ CrewAI integration (optional).

Import :class:`AgentLensCrewAIListener` and attach it to a CrewAI execution to
record tool / LLM / agent / task error activity into an existing AgentLens run.
Requires the optional ``crewai`` extra::

    pip install "agentlens[crewai]"

Nothing in AgentLens core depends on CrewAI; this subpackage is the only place
that imports it.
"""

from __future__ import annotations

try:
    import crewai.events as _crewai_events  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the extra
    raise ModuleNotFoundError(
        "The AgentLens CrewAI integration requires the optional 'crewai' "
        'dependencies. Install them with: pip install "agentlens[crewai]"'
    ) from exc

from agentlens.integrations.crewai.listener import AgentLensCrewAIListener

__all__ = ["AgentLensCrewAIListener"]

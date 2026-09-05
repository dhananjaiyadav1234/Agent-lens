"""AgentLens ↔ OpenAI Agents SDK integration (optional).

Import :class:`AgentLensOpenAITracer` and register it as an OpenAI Agents SDK
tracing processor to record span activity into an existing AgentLens run.
Requires the optional ``openai-agents`` extra::

    pip install "agentlens[openai-agents]"

Nothing in AgentLens core depends on the OpenAI Agents SDK; this subpackage is
the only place that imports it.
"""

from __future__ import annotations

from agentlens.integrations.openai_agents.tracer import AgentLensOpenAITracer

__all__ = ["AgentLensOpenAITracer"]

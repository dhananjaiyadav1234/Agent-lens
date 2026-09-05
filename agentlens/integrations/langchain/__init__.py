"""AgentLens ↔ LangChain integration (optional).

Import :class:`AgentLensCallbackHandler` and attach it to a LangChain execution
to record tool / LLM / error activity into an existing AgentLens run. Requires
the optional ``langchain`` extra::

    pip install "agentlens[langchain]"

Nothing in AgentLens core depends on LangChain; this subpackage is the only
place that imports it.
"""

from __future__ import annotations

from agentlens.integrations.langchain.callback import AgentLensCallbackHandler

__all__ = ["AgentLensCallbackHandler"]

"""AgentLens framework integrations.

Each integration lives in its own subpackage and is optional. This module
imports nothing framework-specific, so ``import agentlens.integrations`` never
pulls in an optional dependency.

Available:

* :mod:`agentlens.integrations.langchain` -- a LangChain callback handler
  (requires the ``langchain`` extra: ``pip install "agentlens[langchain]"``).
* :mod:`agentlens.integrations.openai_agents` -- an OpenAI Agents SDK tracing
  processor (requires the ``openai-agents`` extra:
  ``pip install "agentlens[openai-agents]"``).
"""

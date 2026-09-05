"""A LangChain ``BaseCallbackHandler`` that records activity into an AgentLens run.

The handler observes LangChain's documented public callback hooks and translates
tool / LLM / agent / error activity into existing :class:`~agentlens.models.AgentEvent`
objects on an existing AgentLens run. It performs no I/O of its own, runs no
detectors, builds no reports, and never changes LangChain's exception behaviour.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agentlens import AgentLens, AgentLensError, TraceStateError
from agentlens.models import EventType
from agentlens.models.base import JsonValue

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the extra
    raise ModuleNotFoundError(
        "The AgentLens LangChain integration requires the optional 'langchain' "
        'dependencies. Install them with: pip install "agentlens[langchain]"'
    ) from exc

__all__ = ["AgentLensCallbackHandler"]

_FRAMEWORK = "langchain"
_MAX_JSON_DEPTH = 6


def _json_safe(value: Any, depth: int = 0) -> JsonValue:
    """Reduce a value to JSON-compatible primitives; anything else becomes ``str``."""

    if depth >= _MAX_JSON_DEPTH:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth + 1) for item in value]
    return str(value)


def _error_output(error: BaseException) -> dict[str, JsonValue]:
    return {"exception_type": type(error).__name__, "message": str(error)}


def _name_from_serialized(serialized: Any, fallback: str) -> str:
    if isinstance(serialized, dict):
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        identifier = serialized.get("id")
        if isinstance(identifier, list) and identifier and isinstance(identifier[-1], str):
            return identifier[-1]
    return fallback


def _llm_output(response: Any) -> JsonValue:
    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        texts: list[JsonValue] = []
        for batch in generations:
            if isinstance(batch, list):
                for generation in batch:
                    text = getattr(generation, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
        if texts:
            return {"generations": texts}
    return {"response": _json_safe(response)}


class AgentLensCallbackHandler(BaseCallbackHandler):
    """Record LangChain tool / LLM / agent / error activity into one AgentLens run.

    Use it inside the run's trace context::

        lens = AgentLens()
        with lens.trace("answer the question") as trace:
            handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
            chain.invoke({...}, config={"callbacks": [handler]})

        issues = lens.detect(trace.run_id)      # detection stays explicit
        report = lens.get_report(trace.run_id)  # reporting stays explicit

    The handler:

    * writes every event to ``run_id`` -- it never creates another run;
    * uses only the public ``AgentLens.record_event`` API;
    * never calls ``detect`` / ``run_detectors`` / ``save_issues`` / ``get_report``;
    * re-raises nothing and swallows nothing from the LangChain application.

    Raises :class:`~agentlens.AgentLensError` at construction if ``run_id`` is not a
    known run, and :class:`~agentlens.TraceStateError` if a callback fires while
    that run is not the lens's active trace.
    """

    def __init__(self, *, lens: AgentLens, run_id: UUID) -> None:
        super().__init__()
        if lens.get_run(run_id) is None:
            raise AgentLensError(
                f"cannot attach AgentLensCallbackHandler: no run with id {run_id!r}"
            )
        self._lens = lens
        self._run_id = run_id
        self._pending_tools: dict[UUID, tuple[str, JsonValue | None]] = {}
        self._pending_models: dict[UUID, str] = {}

    @property
    def run_id(self) -> UUID:
        """The AgentLens run every recorded event belongs to."""

        return self._run_id

    # -- internal recording -----------------------------------------

    def _record(
        self,
        *,
        event_type: EventType,
        name: str,
        input: JsonValue | None = None,
        output: JsonValue | None = None,
        status: str | None = None,
        langchain_run_id: UUID | None = None,
    ) -> None:
        active = self._lens.active_trace
        if active is None or active.run_id != self._run_id:
            raise TraceStateError(
                f"AgentLensCallbackHandler for run {self._run_id} requires that run to be the "
                "lens's active trace -- invoke LangChain inside `with lens.trace(...)`"
            )
        metadata: dict[str, JsonValue] = {"framework": _FRAMEWORK}
        if langchain_run_id is not None:
            metadata["langchain_run_id"] = str(langchain_run_id)
        self._lens.record_event(
            event_type=event_type,
            name=name,
            input=input,
            output=output,
            status=status,
            metadata=metadata,
        )

    # -- tools ----------------------------------------------------

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = _name_from_serialized(serialized, "tool")
        inputs = kwargs.get("inputs")
        tool_input: JsonValue = (
            _json_safe(inputs) if isinstance(inputs, dict) else {"input": _json_safe(input_str)}
        )
        self._pending_tools[run_id] = (name, tool_input)
        self._record(
            event_type=EventType.TOOL_CALL_STARTED,
            name=name,
            input=tool_input,
            langchain_run_id=run_id,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name, tool_input = self._pending_tools.pop(run_id, ("tool", None))
        self._record(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name=name,
            input=tool_input,
            output=_json_safe(output),
            status="ok",
            langchain_run_id=run_id,
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name, tool_input = self._pending_tools.pop(run_id, ("tool", None))
        self._record(
            event_type=EventType.ERROR,
            name=type(error).__name__,
            input=tool_input,
            output={"operation": name, **_error_output(error)},
            status="error",
            langchain_run_id=run_id,
        )

    # -- llm / chat model --------------------------------------

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = _name_from_serialized(serialized, "llm")
        self._pending_models[run_id] = name
        self._record(
            event_type=EventType.LLM_CALL_STARTED,
            name=name,
            input={"prompts": _json_safe(prompts)},
            langchain_run_id=run_id,
        )

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = _name_from_serialized(serialized, "chat_model")
        self._pending_models[run_id] = name
        self._record(
            event_type=EventType.LLM_CALL_STARTED,
            name=name,
            input={"messages": _json_safe(messages)},
            langchain_run_id=run_id,
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._pending_models.pop(run_id, "llm")
        self._record(
            event_type=EventType.LLM_CALL_COMPLETED,
            name=name,
            output=_llm_output(response),
            status="ok",
            langchain_run_id=run_id,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._pending_models.pop(run_id, "llm")
        self._record(
            event_type=EventType.ERROR,
            name=type(error).__name__,
            output={"operation": name, **_error_output(error)},
            status="error",
            langchain_run_id=run_id,
        )

    # -- agent / chain ----------------------------------------

    def on_agent_action(self, action: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(
            event_type=EventType.DECISION,
            name="agent_action",
            input={
                "tool": _json_safe(getattr(action, "tool", None)),
                "tool_input": _json_safe(getattr(action, "tool_input", None)),
            },
            output={"log": _json_safe(getattr(action, "log", None))},
            langchain_run_id=run_id,
        )

    def on_agent_finish(self, finish: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(
            event_type=EventType.DECISION,
            name="agent_finish",
            output={"return_values": _json_safe(getattr(finish, "return_values", None))},
            langchain_run_id=run_id,
        )

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(
            event_type=EventType.ERROR,
            name=type(error).__name__,
            output=_error_output(error),
            status="error",
            langchain_run_id=run_id,
        )

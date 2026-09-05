"""An OpenAI Agents SDK ``TracingProcessor`` that records spans into an AgentLens run.

The tracer subscribes to the SDK's public tracing extension point
(``agents.tracing.TracingProcessor`` + ``add_trace_processor``). It adopts the
OpenAI trace(s) created while its AgentLens run is the lens's active trace, and
translates ``function`` / ``generation`` / ``response`` / ``handoff`` spans into
existing :class:`~agentlens.models.AgentEvent` objects via the public
``AgentLens.record_event`` API. It performs no I/O, runs no detectors, builds no
reports, and does not change the SDK's execution behaviour.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agentlens import AgentLens, AgentLensError, TraceStateError
from agentlens.models import EventType
from agentlens.models.base import JsonValue

try:
    from agents.tracing import TracingProcessor
except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the extra
    raise ModuleNotFoundError(
        "The AgentLens OpenAI Agents integration requires the optional "
        "'openai-agents' dependencies. Install them with: "
        'pip install "agentlens[openai-agents]"'
    ) from exc

__all__ = ["AgentLensOpenAITracer"]

_FRAMEWORK = "openai_agents"
_MAX_JSON_DEPTH = 6
_LLM_SPAN_TYPES = frozenset({"generation", "response"})


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


def _tool_input(raw: Any) -> JsonValue | None:
    """Normalise a ``FunctionSpanData.input`` (usually a JSON string) to JSON data."""

    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return _json_safe(json.loads(raw))
        except (ValueError, TypeError):
            return {"input": raw}
    return _json_safe(raw)


def _span_error_parts(error: Any) -> tuple[str, str]:
    """Return ``(exception_type, message)`` for an SDK ``SpanError`` (dict-shaped)."""

    if isinstance(error, dict):
        message = error.get("message")
        data = error.get("data")
    else:
        message = getattr(error, "message", None)
        data = getattr(error, "data", None)
    exception_type = "AgentError"
    if isinstance(data, dict):
        for key in ("exception_type", "error_type", "type"):
            candidate = data.get(key)
            if isinstance(candidate, str) and candidate:
                exception_type = candidate
                break
    return exception_type, str(message) if message is not None else ""


def _error_event_output(operation: str, error: Any) -> dict[str, JsonValue]:
    exception_type, message = _span_error_parts(error)
    return {"operation": operation, "exception_type": exception_type, "message": message}


def _model_name(span_data: Any) -> str:
    model = getattr(span_data, "model", None)
    if isinstance(model, str) and model:
        return model
    response = getattr(span_data, "response", None)
    response_model = getattr(response, "model", None)
    if isinstance(response_model, str) and response_model:
        return response_model
    return "llm"


def _llm_input(span_data: Any) -> JsonValue | None:
    value = getattr(span_data, "input", None)
    return {"input": _json_safe(value)} if value is not None else None


def _llm_output(span_data: Any) -> JsonValue:
    output = getattr(span_data, "output", None)
    if output is not None:
        return {"output": _json_safe(output)}
    response = getattr(span_data, "response", None)
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return {"output_text": text}
    return {"response": _json_safe(response)}


class AgentLensOpenAITracer(TracingProcessor):
    """Records OpenAI Agents SDK span activity into one existing AgentLens run.

    Use it inside the run's trace context::

        lens = AgentLens()
        with lens.trace("answer the question") as trace:
            tracer = AgentLensOpenAITracer(lens=lens, run_id=trace.run_id)
            agents.tracing.add_trace_processor(tracer)
            Runner.run_sync(agent, "...")

        issues = lens.detect(trace.run_id)      # detection stays explicit
        report = lens.get_report(trace.run_id)  # reporting stays explicit

    The tracer:

    * only records spans from OpenAI traces started while ``run_id`` is the lens's
      active trace -- so several tracers registered globally never leak between
      AgentLens runs;
    * writes every event to ``run_id`` through the public ``record_event`` API and
      never creates another run;
    * never calls ``detect`` / ``run_detectors`` / ``save_issues`` / ``get_report``;
    * makes no network or provider calls -- it only reads span data the SDK hands it.

    Raises :class:`~agentlens.AgentLensError` at construction if ``run_id`` is not a
    known run, and :class:`~agentlens.TraceStateError` if a span is recorded while
    that run is not the lens's active trace.
    """

    def __init__(self, *, lens: AgentLens, run_id: UUID) -> None:
        if lens.get_run(run_id) is None:
            raise AgentLensError(f"cannot attach AgentLensOpenAITracer: no run with id {run_id!r}")
        self._lens = lens
        self._run_id = run_id
        self._adopted_trace_ids: set[str] = set()
        self._pending_tools: dict[str, tuple[str, JsonValue | None]] = {}

    @property
    def run_id(self) -> UUID:
        """The AgentLens run every recorded event belongs to."""

        return self._run_id

    # -- TracingProcessor interface ----------------------------------

    def on_trace_start(self, trace: Any) -> None:
        if self._my_run_is_active():
            trace_id = getattr(trace, "trace_id", None)
            if isinstance(trace_id, str):
                self._adopted_trace_ids.add(trace_id)

    def on_trace_end(self, trace: Any) -> None:
        self._adopted_trace_ids.discard(getattr(trace, "trace_id", None))

    def on_span_start(self, span: Any) -> None:
        if getattr(span, "trace_id", None) not in self._adopted_trace_ids:
            return
        data = span.span_data
        kind = getattr(data, "type", None)
        if kind == "function":
            name = _json_safe(getattr(data, "name", None))
            name = name if isinstance(name, str) and name else "tool"
            tool_input = _tool_input(getattr(data, "input", None))
            self._pending_tools[span.span_id] = (name, tool_input)
            self._record(EventType.TOOL_CALL_STARTED, name, span, input=tool_input)
        elif kind in _LLM_SPAN_TYPES:
            self._record(
                EventType.LLM_CALL_STARTED, _model_name(data), span, input=_llm_input(data)
            )

    def on_span_end(self, span: Any) -> None:
        if getattr(span, "trace_id", None) not in self._adopted_trace_ids:
            return
        data = span.span_data
        kind = getattr(data, "type", None)
        error = getattr(span, "error", None)

        if kind == "function":
            name, tool_input = self._pending_tools.pop(span.span_id, ("tool", None))
            if error is not None:
                self._record(
                    EventType.ERROR,
                    _span_error_parts(error)[0],
                    span,
                    input=tool_input,
                    output=_error_event_output(name, error),
                    status="error",
                )
            else:
                self._record(
                    EventType.TOOL_CALL_COMPLETED,
                    name,
                    span,
                    input=tool_input,
                    output=_json_safe(getattr(data, "output", None)),
                    status="ok",
                )
        elif kind in _LLM_SPAN_TYPES:
            name = _model_name(data)
            if error is not None:
                self._record(
                    EventType.ERROR,
                    _span_error_parts(error)[0],
                    span,
                    output=_error_event_output(name, error),
                    status="error",
                )
            else:
                self._record(
                    EventType.LLM_CALL_COMPLETED, name, span, output=_llm_output(data), status="ok"
                )
        elif kind == "handoff":
            self._record(
                EventType.DECISION,
                "handoff",
                span,
                input={
                    "from_agent": _json_safe(getattr(data, "from_agent", None)),
                    "to_agent": _json_safe(getattr(data, "to_agent", None)),
                },
            )

    def shutdown(self) -> None:  # pragma: no cover - nothing to flush
        pass

    def force_flush(self) -> None:  # pragma: no cover - nothing to flush
        pass

    # -- internals -----------------------------------------------

    def _my_run_is_active(self) -> bool:
        active = self._lens.active_trace
        return active is not None and active.run_id == self._run_id

    def _record(
        self,
        event_type: EventType,
        name: str,
        span: Any,
        *,
        input: JsonValue | None = None,
        output: JsonValue | None = None,
        status: str | None = None,
    ) -> None:
        active = self._lens.active_trace
        if active is None or active.run_id != self._run_id:
            raise TraceStateError(
                f"AgentLensOpenAITracer for run {self._run_id} requires that run to be the "
                "lens's active trace -- run the agent inside `with lens.trace(...)`"
            )
        metadata: dict[str, JsonValue] = {"framework": _FRAMEWORK}
        trace_id = getattr(span, "trace_id", None)
        span_id = getattr(span, "span_id", None)
        if isinstance(trace_id, str):
            metadata["framework_run_id"] = trace_id
        if isinstance(span_id, str):
            metadata["framework_span_id"] = span_id
        self._lens.record_event(
            event_type=event_type,
            name=name,
            input=input,
            output=output,
            status=status,
            metadata=metadata,
        )

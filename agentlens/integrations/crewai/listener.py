"""A CrewAI ``BaseEventListener`` that records activity into an AgentLens run.

The listener subscribes to CrewAI's public event bus (``crewai.events`` --
``BaseEventListener`` + the global ``crewai_event_bus``) and translates tool /
LLM / agent / task error activity into existing
:class:`~agentlens.models.AgentEvent` objects on an existing AgentLens run via
the public ``AgentLens.record_event`` API. It performs no I/O of its own, runs
no detectors, builds no reports, and never changes CrewAI's execution behaviour.

CrewAI dispatches synchronous event handlers on a background ``ThreadPoolExecutor``
(``crewai_event_bus`` returns immediately from ``emit``). AgentLens runs and
trace stores are single-threaded, so the listener does not touch the lens from
those worker threads: each handler only appends a JSON-safe payload to an
in-memory buffer (guarded by a lock and by an "is my run active?" check). The
buffered events are replayed into the lens -- in CrewAI emission order -- when
:meth:`AgentLensCrewAIListener.flush` runs on the owning thread, which
:meth:`~AgentLensCrewAIListener.__exit__` does automatically. Using the listener
as a context manager inside ``with lens.trace(...)`` is therefore the supported
pattern.
"""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Any
from uuid import UUID

from agentlens import AgentLens, AgentLensError, TraceStateError
from agentlens.models import EventType
from agentlens.models.base import JsonValue

try:
    from crewai.events import (
        AgentExecutionErrorEvent,
        BaseEventListener,
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
        TaskFailedEvent,
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
        crewai_event_bus,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - only hit without the extra
    raise ModuleNotFoundError(
        "The AgentLens CrewAI integration requires the optional 'crewai' "
        'dependencies. Install them with: pip install "agentlens[crewai]"'
    ) from exc

__all__ = ["AgentLensCrewAIListener"]

_FRAMEWORK = "crewai"
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


def _exception_parts(error: Any, fallback: str) -> tuple[str, str]:
    """Return ``(exception_type, message)`` for an exception, an error string, or None."""

    if isinstance(error, BaseException):
        return type(error).__name__, str(error)
    if isinstance(error, type) and issubclass(error, BaseException):
        return error.__name__, ""
    if error is None:
        return fallback, ""
    return fallback, str(error)


def _tool_name(event: Any) -> str:
    name = getattr(event, "tool_name", None)
    return name if isinstance(name, str) and name else "tool"


def _model_name(event: Any) -> str:
    model = getattr(event, "model", None)
    return model if isinstance(model, str) and model else "llm"


def _llm_input(event: Any) -> JsonValue | None:
    messages = getattr(event, "messages", None)
    return {"messages": _json_safe(messages)} if messages is not None else None


class AgentLensCrewAIListener(BaseEventListener):
    """Record CrewAI tool / LLM / agent / task error activity into one AgentLens run.

    Use it as a context manager inside the run's trace context::

        lens = AgentLens()
        with lens.trace("research the topic") as trace:
            with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
                crew.kickoff()

        issues = lens.detect(trace.run_id)      # detection stays explicit
        report = lens.get_report(trace.run_id)  # reporting stays explicit

    The listener:

    * writes every event to ``run_id`` -- it never creates another run or trace;
    * uses only the public ``AgentLens.record_event`` API;
    * never calls ``detect`` / ``run_detectors`` / ``save_issues`` / ``get_report``;
    * only records activity emitted while ``run_id`` is the lens's active trace,
      so several listeners registered on the global bus never leak between runs;
    * makes no network or provider calls -- it only reads data CrewAI hands it.

    CrewAI has no public "agent decided X" event in the tested version, so no
    :class:`~agentlens.models.EventType` ``DECISION`` events are produced; this is
    intentional rather than an omission.

    Raises :class:`~agentlens.AgentLensError` at construction if ``run_id`` is not
    a known run, and :class:`~agentlens.TraceStateError` from :meth:`flush` if
    buffered events exist while ``run_id`` is not the lens's active trace.
    """

    def __init__(self, *, lens: AgentLens, run_id: UUID) -> None:
        if lens.get_run(run_id) is None:
            raise AgentLensError(
                f"cannot attach AgentLensCrewAIListener: no run with id {run_id!r}"
            )
        self._lens = lens
        self._run_id = run_id
        self._buffer: list[tuple[int, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._registered: list[tuple[type, Any]] = []
        super().__init__()

    @property
    def run_id(self) -> UUID:
        """The AgentLens run every recorded event belongs to."""

        return self._run_id

    # -- BaseEventListener interface --------------------------------

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        handlers = (
            (ToolUsageStartedEvent, self._on_tool_started),
            (ToolUsageFinishedEvent, self._on_tool_finished),
            (ToolUsageErrorEvent, self._on_tool_error),
            (LLMCallStartedEvent, self._on_llm_started),
            (LLMCallCompletedEvent, self._on_llm_completed),
            (LLMCallFailedEvent, self._on_llm_failed),
            (AgentExecutionErrorEvent, self._on_agent_error),
            (TaskFailedEvent, self._on_task_failed),
        )
        for event_type, handler in handlers:
            crewai_event_bus.on(event_type)(handler)
            self._registered.append((event_type, handler))

    # -- lifecycle ---------------------------------------------------

    def __enter__(self) -> AgentLensCrewAIListener:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self.flush()
        finally:
            self.detach()

    def detach(self) -> None:
        """Unregister the listener's handlers from the global CrewAI event bus."""

        for event_type, handler in self._registered:
            crewai_event_bus.off(event_type, handler)
        self._registered.clear()

    def flush(self, *, handler_timeout: float | None = 30.0) -> int:
        """Replay buffered CrewAI events into the AgentLens run, in emission order.

        Call this on the thread that owns the lens, while ``run_id`` is the active
        trace (``__exit__`` does so automatically). Returns the number of events
        recorded. Waits up to ``handler_timeout`` seconds for CrewAI's own worker
        threads to finish delivering events before draining the buffer.
        """

        crewai_event_bus.flush(timeout=handler_timeout)
        with self._lock:
            pending = sorted(self._buffer, key=lambda item: item[0])
            self._buffer.clear()
        if not pending:
            return 0
        active = self._lens.active_trace
        if active is None or active.run_id != self._run_id:
            raise TraceStateError(
                f"AgentLensCrewAIListener for run {self._run_id} requires that run to be the "
                "lens's active trace when flushing -- run the crew and flush inside "
                "`with lens.trace(...)`"
            )
        for _seq, kwargs in pending:
            self._lens.record_event(**kwargs)
        return len(pending)

    # -- internal buffering ----------------------------------------

    def _my_run_is_active(self) -> bool:
        active = self._lens.active_trace
        return active is not None and active.run_id == self._run_id

    def _buffer_event(self, event: Any, **kwargs: Any) -> None:
        # CrewAI dispatches handlers on worker threads. Reading the lens's active
        # trace here is a best-effort guard: it keeps events emitted while a
        # *different* AgentLens run is active out of this listener's buffer, so
        # listeners sharing the global bus do not cross-contaminate. Events for
        # this run are only ever written to the lens later, from `flush`, on the
        # owning thread.
        if not self._my_run_is_active():
            return
        sequence = getattr(event, "emission_sequence", 0) or 0
        kwargs["metadata"] = self._metadata(event)
        with self._lock:
            self._buffer.append((int(sequence), kwargs))

    def _metadata(self, event: Any) -> dict[str, JsonValue]:
        meta: dict[str, JsonValue] = {"framework": _FRAMEWORK}
        event_type = getattr(event, "type", None)
        if isinstance(event_type, str) and event_type:
            meta["framework_event_type"] = event_type
        for key, attr in (
            ("framework_event_id", "event_id"),
            ("framework_task_id", "task_id"),
            ("framework_agent_id", "agent_id"),
        ):
            value = getattr(event, attr, None)
            if value is not None:
                meta[key] = str(value)
        return meta

    # -- tool events ----------------------------------------------

    def _on_tool_started(self, _source: Any, event: Any) -> None:
        self._buffer_event(
            event,
            event_type=EventType.TOOL_CALL_STARTED,
            name=_tool_name(event),
            input=_json_safe(getattr(event, "tool_args", None)),
        )

    def _on_tool_finished(self, _source: Any, event: Any) -> None:
        failure = getattr(event, "failure", None)
        if failure is not None:
            # The call returned but reported that it did not do the work. Record
            # it as an error, never as a (synthesised) completion.
            message = getattr(failure, "message", None) or str(failure)
            self._buffer_event(
                event,
                event_type=EventType.ERROR,
                name="ToolFailure",
                input=_json_safe(getattr(event, "tool_args", None)),
                output={
                    "operation": _tool_name(event),
                    "exception_type": "ToolFailure",
                    "message": str(message),
                },
                status="error",
            )
            return
        self._buffer_event(
            event,
            event_type=EventType.TOOL_CALL_COMPLETED,
            name=_tool_name(event),
            input=_json_safe(getattr(event, "tool_args", None)),
            output=_json_safe(getattr(event, "output", None)),
            status="ok",
        )

    def _on_tool_error(self, _source: Any, event: Any) -> None:
        exc_type, message = _exception_parts(getattr(event, "error", None), "ToolUsageError")
        self._buffer_event(
            event,
            event_type=EventType.ERROR,
            name=exc_type,
            input=_json_safe(getattr(event, "tool_args", None)),
            output={
                "operation": _tool_name(event),
                "exception_type": exc_type,
                "message": message,
            },
            status="error",
        )

    # -- llm events ----------------------------------------------

    def _on_llm_started(self, _source: Any, event: Any) -> None:
        self._buffer_event(
            event,
            event_type=EventType.LLM_CALL_STARTED,
            name=_model_name(event),
            input=_llm_input(event),
        )

    def _on_llm_completed(self, _source: Any, event: Any) -> None:
        self._buffer_event(
            event,
            event_type=EventType.LLM_CALL_COMPLETED,
            name=_model_name(event),
            output={"response": _json_safe(getattr(event, "response", None))},
            status="ok",
        )

    def _on_llm_failed(self, _source: Any, event: Any) -> None:
        exc_type, message = _exception_parts(getattr(event, "error", None), "LLMCallError")
        self._buffer_event(
            event,
            event_type=EventType.ERROR,
            name=exc_type,
            output={
                "operation": _model_name(event),
                "exception_type": exc_type,
                "message": message,
            },
            status="error",
        )

    # -- agent / task errors ------------------------------------

    def _on_agent_error(self, _source: Any, event: Any) -> None:
        exc_type, message = _exception_parts(getattr(event, "error", None), "AgentExecutionError")
        self._buffer_event(
            event,
            event_type=EventType.ERROR,
            name=exc_type,
            output={
                "operation": "agent_execution",
                "exception_type": exc_type,
                "message": message,
            },
            status="error",
        )

    def _on_task_failed(self, _source: Any, event: Any) -> None:
        exc_type, message = _exception_parts(getattr(event, "error", None), "TaskError")
        raw_type = getattr(event, "error_type", None)
        if isinstance(raw_type, type) and issubclass(raw_type, BaseException):
            exc_type = raw_type.__name__
        self._buffer_event(
            event,
            event_type=EventType.ERROR,
            name=exc_type,
            output={"operation": "task", "exception_type": exc_type, "message": message},
            status="error",
        )

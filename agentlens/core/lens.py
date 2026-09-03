"""The ``AgentLens`` entry point and its ``trace(...)`` context manager."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from agentlens.core.errors import TraceStateError
from agentlens.core.run_manager import RunManager
from agentlens.core.storage import InMemoryTraceStore
from agentlens.core.tracer import Trace
from agentlens.models import AgentEvent, AgentRun, EventType
from agentlens.models.base import JsonMapping, JsonValue

__all__ = ["AgentLens"]


def _error_info(exc: BaseException) -> JsonMapping:
    """Reduce an exception to safe, JSON-compatible fields.

    Only the type name and the ``str()`` of the exception are kept. No traceback,
    no ``args`` tuple, no exception object -- the universal event model must stay
    JSON-serialisable and framework-agnostic.
    """

    return {"exception_type": type(exc).__name__, "message": str(exc)}


class AgentLens:
    """Local tracing engine for a single process / thread of control.

    Each instance owns its own in-memory store, run manager and active-trace
    slot, so multiple ``AgentLens`` instances never share runs, events or trace
    state.

    Not thread-safe and not async-context-aware: the active trace is plain
    instance state. One instance is meant to be driven by one synchronous flow at
    a time. Nested traces are rejected (see :meth:`trace`).
    """

    def __init__(self) -> None:
        self._store = InMemoryTraceStore()
        self._run_manager = RunManager(self._store)
        self._active_trace: Trace | None = None

    # -- active state --------------------------------------------------

    @property
    def active_trace(self) -> Trace | None:
        """The currently open :class:`Trace`, or ``None`` if no trace is active."""

        return self._active_trace

    # -- tracing ------------------------------------------------------

    @contextmanager
    def trace(self, task: str, *, metadata: dict[str, Any] | None = None) -> Iterator[Trace]:
        """Open a trace for ``task``: start a run, yield a recorder, close it out.

        On normal exit the run transitions to ``SUCCESS`` and a ``RUN_COMPLETED``
        event is recorded. If the body raises, an ``ERROR`` event is recorded, the
        run transitions to ``FAILED``, and the exception is re-raised unchanged.

        Nested traces on the same instance raise :class:`TraceStateError`.
        """

        if self._active_trace is not None:
            raise TraceStateError(
                "a trace is already active on this AgentLens instance; nested "
                "traces are not supported -- close the current trace first, or "
                "use a separate AgentLens instance"
            )

        run = self._run_manager.start_run(task, metadata=metadata)
        handle = Trace(run, self._store)
        handle._emit(event_type=EventType.RUN_STARTED, name="run_started")
        self._active_trace = handle

        try:
            try:
                yield handle
            except BaseException as exc:
                handle._emit(
                    event_type=EventType.ERROR,
                    name=type(exc).__name__,
                    output=_error_info(exc),
                    status="error",
                )
                handle._close()
                self._run_manager.fail_run(run.id)
                raise
            else:
                handle._emit(event_type=EventType.RUN_COMPLETED, name="run_completed")
                handle._close()
                self._run_manager.complete_run(run.id)
        finally:
            self._active_trace = None

    def record_event(
        self,
        *,
        event_type: EventType,
        name: str,
        input: JsonValue | None = None,
        output: JsonValue | None = None,
        duration_ms: float | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Convenience: record an event on the active trace.

        Raises :class:`TraceStateError` if there is no active trace.
        """

        if self._active_trace is None:
            raise TraceStateError(
                "no active trace: open one with `with lens.trace(...) as trace:` "
                "before recording events"
            )
        return self._active_trace.record_event(
            event_type=event_type,
            name=name,
            input=input,
            output=output,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata,
        )

    # -- read access ------------------------------------------------

    def get_run(self, run_id: UUID) -> AgentRun | None:
        return self._store.get_run(run_id)

    def get_events(self, run_id: UUID) -> list[AgentEvent]:
        return self._store.get_events(run_id)

    def list_runs(self) -> list[AgentRun]:
        return self._store.list_runs()

    def list_completed_runs(self) -> list[AgentRun]:
        return self._store.list_completed_runs()

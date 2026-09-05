"""The ``AgentLens`` entry point and its ``trace(...)`` context manager."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from agentlens.core.errors import AgentLensError, TraceStateError
from agentlens.core.run_manager import RunManager
from agentlens.core.storage import InMemoryTraceStore, TraceStore
from agentlens.core.tracer import Trace
from agentlens.detectors.orchestration import run_detectors
from agentlens.models import AgentEvent, AgentIssue, AgentReport, AgentRun, EventType
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

    By default an instance keeps everything in memory. Pass ``store=`` a
    :class:`~agentlens.core.storage.TraceStore` (for example
    :class:`agentlens.storage.SQLiteTraceStore`) to persist runs and events.
    """

    def __init__(self, *, store: TraceStore | None = None) -> None:
        self._store: TraceStore = store if store is not None else InMemoryTraceStore()
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

    # -- detection --------------------------------------------------

    def detect(self, run: AgentRun | UUID) -> list[AgentIssue]:
        """Run every deterministic detector over one stored run, persist and return the issues.

        Accepts a run id or an :class:`~agentlens.models.AgentRun`; either way the
        run and its events are looked up in this instance's store, so only that
        run's events are analysed. Detectors run in a fixed order
        (loop, retry, duplicate-tool, inefficiency) with their default
        configuration -- see :func:`agentlens.detectors.run_detectors`.

        The generated issues are handed to ``store.save_issues`` and then returned
        unchanged. Persistence is **append-only and not de-duplicated**: calling
        ``detect`` again on the same run appends a second batch of issue records
        (each with its own fresh ``id``). Use :meth:`get_issues` to read back what
        has been persisted without re-running detection.

        The run and its events are never modified. Raises :class:`AgentLensError`
        if no run with that id is stored; a failure inside ``save_issues``
        propagates.
        """

        run_id = run.id if isinstance(run, AgentRun) else run
        stored_run = self._store.get_run(run_id)
        if stored_run is None:
            raise AgentLensError(f"cannot detect issues: no run with id {run_id!r}")
        events = self._store.get_events(run_id)
        issues = run_detectors(stored_run, events)
        self._store.save_issues(issues)
        return issues

    def get_issues(self, run_id: UUID) -> list[AgentIssue]:
        """Return issues already persisted for ``run_id`` (a fresh list).

        This is a plain read: it never runs detectors. An unknown run id, or a
        run for which :meth:`detect` has not been called, returns ``[]``.
        """

        return self._store.get_issues(run_id)

    def get_report(self, run_id: UUID) -> AgentReport:
        """Return a read-only aggregate :class:`~agentlens.models.AgentReport` for ``run_id``.

        Retrieves the stored run, its events (in ``sequence_number`` order), and
        its already-persisted issues (in save order), then computes a
        deterministic :class:`~agentlens.models.ReportSummary` over them.

        This is purely a read: it does **not** run detectors, call :meth:`detect`,
        create issues, or write anything to storage. Repeated calls on unchanged
        data return byte-identical content. Raises :class:`AgentLensError` if no
        run with that id is stored -- consistent with :meth:`detect`.
        """

        run = self._store.get_run(run_id)
        if run is None:
            raise AgentLensError(f"cannot build report: no run with id {run_id!r}")
        events = self._store.get_events(run_id)
        issues = self._store.get_issues(run_id)
        return AgentReport.build(run, events, issues)

    # -- read access ------------------------------------------------

    def get_run(self, run_id: UUID) -> AgentRun | None:
        return self._store.get_run(run_id)

    def get_events(self, run_id: UUID) -> list[AgentEvent]:
        return self._store.get_events(run_id)

    def list_runs(self) -> list[AgentRun]:
        return self._store.list_runs()

    def list_completed_runs(self) -> list[AgentRun]:
        return self._store.list_completed_runs()

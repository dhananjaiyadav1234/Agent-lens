"""The trace handle yielded by ``AgentLens.trace(...)``.

A :class:`Trace` is a thin recorder bound to one run. It assigns sequence numbers
and UTC timestamps, builds validated :class:`~agentlens.models.AgentEvent`
objects, and writes them to the store. Lifecycle transitions are *not* its job --
``AgentLens`` drives those via :class:`~agentlens.core.run_manager.RunManager`.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agentlens.core.errors import TraceStateError
from agentlens.core.storage import InMemoryTraceStore
from agentlens.models import AgentEvent, AgentRun, EventType
from agentlens.models.base import JsonValue

__all__ = ["Trace"]

_FIRST_SEQUENCE_NUMBER = 0


class Trace:
    """Records ordered events for a single active run.

    Instances are created by :class:`~agentlens.core.lens.AgentLens`; application
    code only ever receives one from the ``with lens.trace(...) as trace`` block
    and calls :meth:`record_event` on it.
    """

    def __init__(self, run: AgentRun, store: InMemoryTraceStore) -> None:
        self._run_id = run.id
        self._store = store
        self._next_sequence = _FIRST_SEQUENCE_NUMBER
        self._closed = False

    # -- identity / state ------------------------------------------------

    @property
    def run_id(self) -> UUID:
        return self._run_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def run(self) -> AgentRun:
        """The current stored state of this trace's run."""

        run = self._store.get_run(self._run_id)
        if run is None:  # pragma: no cover - store is never pruned in this milestone
            raise TraceStateError(f"run {self._run_id} is no longer in the store")
        return run

    def events(self) -> list[AgentEvent]:
        """All events recorded for this run so far, ordered by sequence number."""

        return self._store.get_events(self._run_id)

    # -- recording ------------------------------------------------------

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
        """Record one event against the active run.

        ``run_id``, ``sequence_number`` and ``timestamp`` are assigned
        automatically. Raises :class:`TraceStateError` if the trace has closed.
        """

        if self._closed:
            raise TraceStateError(
                f"cannot record '{name}' event: the trace for run {self._run_id} has already closed"
            )
        return self._emit(
            event_type=event_type,
            name=name,
            input=input,
            output=output,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata,
        )

    # -- internals (used by AgentLens) ---------------------------------

    def _emit(
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
        """Build, store and return an event. Does not check ``_closed``.

        Used for the framework-emitted ``RUN_STARTED`` / ``RUN_COMPLETED`` /
        ``ERROR`` events, which are recorded while the trace is still open.
        """

        event = AgentEvent(
            run_id=self._run_id,
            sequence_number=self._next_sequence,
            event_type=event_type,
            name=name,
            input=input,
            output=output,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata or {},
        )
        self._next_sequence += 1
        self._store.add_event(event)
        return event

    def _close(self) -> None:
        self._closed = True

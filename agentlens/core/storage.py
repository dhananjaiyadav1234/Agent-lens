"""In-memory storage for runs and events.

This milestone keeps everything in process memory. The tracing engine talks to
storage only through the small surface defined here, so a persistent backend
(SQLite/SQLAlchemy) can be introduced later without changing the public tracing
API. Nothing in this module is thread-safe.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from agentlens.models import AgentEvent, AgentRun, RunStatus

__all__ = ["InMemoryTraceStore"]


class InMemoryTraceStore:
    """Holds runs and their events in memory for the lifetime of the process.

    Runs are keyed by id; events are appended per run in recording order. Reads
    return copies of the internal containers so callers cannot mutate stored
    state by accident.
    """

    def __init__(self) -> None:
        self._runs: dict[UUID, AgentRun] = {}
        self._events: dict[UUID, list[AgentEvent]] = defaultdict(list)

    # -- runs ---------------------------------------------------------------

    def save_run(self, run: AgentRun) -> None:
        """Insert or replace a run (keyed by ``run.id``).

        Replacement is how atomic lifecycle transitions are stored: the caller
        builds a fully validated ``AgentRun`` for the next state and hands it in.
        """

        self._runs[run.id] = run

    def get_run(self, run_id: UUID) -> AgentRun | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[AgentRun]:
        """All runs, in insertion order."""

        return list(self._runs.values())

    def list_completed_runs(self) -> list[AgentRun]:
        """Runs that have reached a terminal state (``SUCCESS`` or ``FAILED``)."""

        return [r for r in self._runs.values() if r.status is not RunStatus.RUNNING]

    # -- events -----------------------------------------------------------

    def add_event(self, event: AgentEvent) -> None:
        self._events[event.run_id].append(event)

    def get_events(self, run_id: UUID) -> list[AgentEvent]:
        """Events for a run, ordered by ``sequence_number`` (a copy)."""

        return sorted(self._events.get(run_id, []), key=lambda e: e.sequence_number)

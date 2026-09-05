"""Trace storage: the ``TraceStore`` contract and its in-memory implementation.

The tracing engine and ``AgentLens`` talk to storage only through the six
methods on :class:`TraceStore`. :class:`InMemoryTraceStore` is the default,
process-memory implementation; :class:`agentlens.storage.SQLiteTraceStore` is a
persistent one. Nothing in this module is thread-safe.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from agentlens.models import AgentEvent, AgentIssue, AgentRun, RunStatus

__all__ = ["TraceStore", "InMemoryTraceStore"]


@runtime_checkable
class TraceStore(Protocol):
    """The storage surface the tracing engine and ``AgentLens`` depend on.

    An implementation must:

    * ``save_run`` -- insert or replace a run keyed by ``run.id`` (run lifecycle
      transitions are stored as full-replacement models);
    * ``get_run`` -- return the stored run or ``None``;
    * ``add_event`` -- append one event (append-only);
    * ``get_events`` -- return a run's events ordered by ``sequence_number``
      ascending, as a fresh list, isolated per ``run_id``;
    * ``list_runs`` / ``list_completed_runs`` -- all runs / terminal runs;
    * ``save_issues`` -- append detected issues (append-only; no dedup, no
      replacement of earlier detection results);
    * ``get_issues`` -- issues for one run, in save order, as a fresh list,
      isolated per ``run_id`` (unknown run -> ``[]``);
    * ``list_issues`` -- all issues, in first-save order.

    Reads must not hand back references that let a caller mutate stored state,
    and must never mix events or issues between runs.
    """

    def save_run(self, run: AgentRun) -> None: ...

    def get_run(self, run_id: UUID) -> AgentRun | None: ...

    def add_event(self, event: AgentEvent) -> None: ...

    def get_events(self, run_id: UUID) -> list[AgentEvent]: ...

    def list_runs(self) -> list[AgentRun]: ...

    def list_completed_runs(self) -> list[AgentRun]: ...

    def save_issues(self, issues: Sequence[AgentIssue]) -> None: ...

    def get_issues(self, run_id: UUID) -> list[AgentIssue]: ...

    def list_issues(self) -> list[AgentIssue]: ...


class InMemoryTraceStore:
    """Holds runs, events, and issues in memory for the lifetime of the process.

    Runs are keyed by id; events and issues are appended per run in the order
    they arrive. Reads return copies of the internal containers so callers
    cannot mutate stored state by accident.
    """

    def __init__(self) -> None:
        self._runs: dict[UUID, AgentRun] = {}
        self._events: dict[UUID, list[AgentEvent]] = defaultdict(list)
        self._issues_by_run: dict[UUID, list[AgentIssue]] = defaultdict(list)
        self._issue_log: list[AgentIssue] = []

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

    # -- issues ---------------------------------------------------------

    def save_issues(self, issues: Sequence[AgentIssue]) -> None:
        """Append issues in the order received. Empty input is a no-op.

        Append-only: issues are never mutated or replaced. Repeated detection of
        the same run appends further records (each with its own fresh ``id``).
        """

        for issue in issues:
            self._issues_by_run[issue.run_id].append(issue)
            self._issue_log.append(issue)

    def get_issues(self, run_id: UUID) -> list[AgentIssue]:
        """Issues for one run, in save order (a fresh list). Unknown run -> ``[]``."""

        return list(self._issues_by_run.get(run_id, []))

    def list_issues(self) -> list[AgentIssue]:
        """All issues, in first-save order (a fresh list)."""

        return list(self._issue_log)

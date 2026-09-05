"""Run lifecycle management.

``RunManager`` owns the *rules* for how an :class:`~agentlens.models.AgentRun`
moves between states. It never mutates a stored run in place: every transition
builds a brand-new, fully validated ``AgentRun`` for the next state and replaces
the stored one. That keeps the model's ``validate_assignment=True`` invariants
satisfied at all times -- there is no window in which ``status`` and
``finished_at`` disagree.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from agentlens.core.errors import RunLifecycleError
from agentlens.core.storage import TraceStore
from agentlens.models import AgentRun, RunStatus
from agentlens.models.base import utcnow

__all__ = ["RunManager"]

_TERMINAL_STATUSES = (RunStatus.SUCCESS, RunStatus.FAILED)


class RunManager:
    """Starts runs and performs atomic terminal transitions."""

    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def start_run(self, task: str, *, metadata: dict | None = None) -> AgentRun:
        """Create and store a new ``RUNNING`` run with a UTC-aware ``started_at``."""

        run = AgentRun(
            task=task,
            status=RunStatus.RUNNING,
            metadata=metadata or {},
        )
        self._store.save_run(run)
        return run

    def complete_run(self, run_id: UUID, *, finished_at: datetime | None = None) -> AgentRun:
        """Transition a ``RUNNING`` run to ``SUCCESS`` atomically."""

        return self._transition(run_id, RunStatus.SUCCESS, finished_at)

    def fail_run(self, run_id: UUID, *, finished_at: datetime | None = None) -> AgentRun:
        """Transition a ``RUNNING`` run to ``FAILED`` atomically."""

        return self._transition(run_id, RunStatus.FAILED, finished_at)

    # -- internals --------------------------------------------------------

    def _transition(
        self,
        run_id: UUID,
        target: RunStatus,
        finished_at: datetime | None,
    ) -> AgentRun:
        current = self._store.get_run(run_id)
        if current is None:
            raise RunLifecycleError(f"cannot transition unknown run {run_id}")
        if current.status in _TERMINAL_STATUSES:
            raise RunLifecycleError(
                f"run {run_id} is already {current.status.name}; "
                f"it cannot be moved to {target.name}"
            )

        # Build the next state as one fully validated instance: status and
        # finished_at are set together, never through separate assignments.
        next_run = AgentRun(
            id=current.id,
            task=current.task,
            status=target,
            started_at=current.started_at,
            finished_at=finished_at or utcnow(),
            metadata=current.metadata,
        )
        self._store.save_run(next_run)
        return next_run

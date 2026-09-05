"""A persistent :class:`~agentlens.core.storage.TraceStore` backed by SQLite.

Uses only the standard-library ``sqlite3`` module. Runs and events are stored
one row each; the JSON-compatible model fields (``metadata`` / ``input`` /
``output``) are kept as JSON text (via ``model_dump(mode="json")``), everything
else in typed columns. On read, rows are fed back through Pydantic validation,
so a retrieved ``AgentRun`` / ``AgentEvent`` is semantically identical to the
one stored.

Runs, events, and the issues produced by ``AgentLens.detect`` are all persisted.
Issues are append-only: repeated detection appends further records rather than
replacing earlier ones, and there is no semantic de-duplication. Nothing here is
thread-safe, and the database path must always be supplied explicitly: there is
no environment or global default.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from uuid import UUID

from agentlens.models import AgentEvent, AgentIssue, AgentRun

__all__ = ["SQLiteTraceStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    metadata    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    name            TEXT NOT NULL,
    input           TEXT NOT NULL,
    output          TEXT NOT NULL,
    duration_ms     REAL,
    status          TEXT,
    metadata        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events (run_id, sequence_number);

-- Issues are append-only: no ``id`` primary key, so re-persisting the same
-- issue record produces a second row (matching InMemoryTraceStore). Row order
-- (SQLite ``rowid``) is the save order.
CREATE TABLE IF NOT EXISTS issues (
    id                TEXT NOT NULL,
    run_id            TEXT NOT NULL,
    issue_type        TEXT NOT NULL,
    severity          TEXT NOT NULL,
    description       TEXT NOT NULL,
    metadata          TEXT NOT NULL,
    related_event_ids TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_issues_run ON issues (run_id);
"""


class SQLiteTraceStore:
    """Stores runs and events in a SQLite database file.

    Parameters
    ----------
    database_path:
        Path to the SQLite database file. It (and its schema) is created if it
        does not exist. Use ``":memory:"`` for an ephemeral database. Passing the
        same path to a new instance later re-opens the same data.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._path = str(database_path)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    # -- lifecycle --------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection. Safe to call more than once."""

        self._connection.close()

    def __enter__(self) -> SQLiteTraceStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- runs -----------------------------------------------------

    def save_run(self, run: AgentRun) -> None:
        """Insert or replace a run, keyed by ``run.id`` (updates keep row order)."""

        data = run.model_dump(mode="json")
        self._connection.execute(
            """
            INSERT INTO runs (id, task, status, started_at, finished_at, metadata)
            VALUES (:id, :task, :status, :started_at, :finished_at, :metadata)
            ON CONFLICT(id) DO UPDATE SET
                task = excluded.task,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                metadata = excluded.metadata
            """,
            {
                "id": data["id"],
                "task": data["task"],
                "status": data["status"],
                "started_at": data["started_at"],
                "finished_at": data["finished_at"],
                "metadata": json.dumps(data["metadata"]),
            },
        )
        self._connection.commit()

    def get_run(self, run_id: UUID) -> AgentRun | None:
        row = self._connection.execute("SELECT * FROM runs WHERE id = ?", (str(run_id),)).fetchone()
        return _row_to_run(row) if row is not None else None

    def list_runs(self) -> list[AgentRun]:
        """All runs, in the order they were first saved."""

        rows = self._connection.execute("SELECT * FROM runs ORDER BY rowid").fetchall()
        return [_row_to_run(row) for row in rows]

    def list_completed_runs(self) -> list[AgentRun]:
        """Runs that have reached a terminal state (``SUCCESS`` or ``FAILED``)."""

        rows = self._connection.execute(
            "SELECT * FROM runs WHERE status != 'running' ORDER BY rowid"
        ).fetchall()
        return [_row_to_run(row) for row in rows]

    # -- events ---------------------------------------------------

    def add_event(self, event: AgentEvent) -> None:
        data = event.model_dump(mode="json")
        self._connection.execute(
            """
            INSERT INTO events (
                id, run_id, sequence_number, timestamp, event_type, name,
                input, output, duration_ms, status, metadata
            )
            VALUES (
                :id, :run_id, :sequence_number, :timestamp, :event_type, :name,
                :input, :output, :duration_ms, :status, :metadata
            )
            """,
            {
                "id": data["id"],
                "run_id": data["run_id"],
                "sequence_number": data["sequence_number"],
                "timestamp": data["timestamp"],
                "event_type": data["event_type"],
                "name": data["name"],
                "input": json.dumps(data["input"]),
                "output": json.dumps(data["output"]),
                "duration_ms": data["duration_ms"],
                "status": data["status"],
                "metadata": json.dumps(data["metadata"]),
            },
        )
        self._connection.commit()

    def get_events(self, run_id: UUID) -> list[AgentEvent]:
        """Events for one run, ordered by ``sequence_number`` then insertion order."""

        rows = self._connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY sequence_number, rowid",
            (str(run_id),),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    # -- issues -------------------------------------------------

    def save_issues(self, issues: Sequence[AgentIssue]) -> None:
        """Append detected issues (append-only; no dedup). Empty input is a no-op."""

        for issue in issues:
            data = issue.model_dump(mode="json")
            self._connection.execute(
                """
                INSERT INTO issues (
                    id, run_id, issue_type, severity, description, metadata,
                    related_event_ids
                )
                VALUES (
                    :id, :run_id, :issue_type, :severity, :description, :metadata,
                    :related_event_ids
                )
                """,
                {
                    "id": data["id"],
                    "run_id": data["run_id"],
                    "issue_type": data["issue_type"],
                    "severity": data["severity"],
                    "description": data["description"],
                    "metadata": json.dumps(data["metadata"]),
                    "related_event_ids": json.dumps(data["related_event_ids"]),
                },
            )
        self._connection.commit()

    def get_issues(self, run_id: UUID) -> list[AgentIssue]:
        """Issues for one run, in save order. Unknown run -> ``[]``."""

        rows = self._connection.execute(
            "SELECT * FROM issues WHERE run_id = ? ORDER BY rowid",
            (str(run_id),),
        ).fetchall()
        return [_row_to_issue(row) for row in rows]

    def list_issues(self) -> list[AgentIssue]:
        """All issues, in first-save order."""

        rows = self._connection.execute("SELECT * FROM issues ORDER BY rowid").fetchall()
        return [_row_to_issue(row) for row in rows]


def _json_columns(row: sqlite3.Row, *names: str) -> Iterator[object]:
    for name in names:
        yield json.loads(row[name])


def _row_to_run(row: sqlite3.Row) -> AgentRun:
    (metadata,) = _json_columns(row, "metadata")
    return AgentRun.model_validate(
        {
            "id": row["id"],
            "task": row["task"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "metadata": metadata,
        }
    )


def _row_to_event(row: sqlite3.Row) -> AgentEvent:
    input_, output, metadata = _json_columns(row, "input", "output", "metadata")
    return AgentEvent.model_validate(
        {
            "id": row["id"],
            "run_id": row["run_id"],
            "sequence_number": row["sequence_number"],
            "timestamp": row["timestamp"],
            "event_type": row["event_type"],
            "name": row["name"],
            "input": input_,
            "output": output,
            "duration_ms": row["duration_ms"],
            "status": row["status"],
            "metadata": metadata,
        }
    )


def _row_to_issue(row: sqlite3.Row) -> AgentIssue:
    metadata, related_event_ids = _json_columns(row, "metadata", "related_event_ids")
    return AgentIssue.model_validate(
        {
            "id": row["id"],
            "run_id": row["run_id"],
            "issue_type": row["issue_type"],
            "severity": row["severity"],
            "description": row["description"],
            "metadata": metadata,
            "related_event_ids": related_event_ids,
        }
    )

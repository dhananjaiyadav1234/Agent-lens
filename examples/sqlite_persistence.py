"""Trace a run into SQLite, reopen the database, and detect issues.

Fully offline and deterministic. Uses a throwaway temp-directory database that is
cleaned up on exit -- no repository artifacts are left behind.

Usage::

    python -m examples.sqlite_persistence
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agentlens import AgentLens
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import run_inefficient_agent


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlens-sqlite-"))
    db_path = workdir / "agentlens.db"
    try:
        # 1-2. write a trace through a SQLite-backed AgentLens
        write_store = SQLiteTraceStore(db_path)
        write_lens = AgentLens(store=write_store)
        run_id = run_inefficient_agent(write_lens)
        write_store.close()
        print(f"wrote run {run_id} to {db_path.name}, then closed the store")

        # 3-5. reopen the same file with a fresh store / lens and analyse
        read_store = SQLiteTraceStore(db_path)
        read_lens = AgentLens(store=read_store)
        run = read_lens.get_run(run_id)
        events = read_lens.get_events(run_id)
        issues = read_lens.detect(run_id)

        print(f"reopened: run status={run.status.value}, {len(events)} events")
        print("event sequence:", [e.sequence_number for e in events])
        for issue in issues:
            print(
                f"  {issue.issue_type.value} / {issue.severity.value} / "
                f"operation={issue.metadata['operation']} "
                f"evidence={issue.metadata['evidence']} "
                f"related_events={len(issue.related_event_ids)}"
            )
        read_store.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

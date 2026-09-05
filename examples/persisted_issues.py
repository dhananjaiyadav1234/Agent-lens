"""Detect issues into SQLite, reopen the database, and read the issues back.

Fully offline and deterministic. Uses a throwaway temp-directory database that is
removed on exit -- no repository artifacts are left behind.

Usage::

    python -m examples.persisted_issues
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agentlens import AgentLens
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import run_inefficient_agent


def _describe(issue) -> str:
    return (
        f"{issue.issue_type.value} / {issue.severity.value} / {issue.description} "
        f"(related_events={len(issue.related_event_ids)})"
    )


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlens-issues-"))
    db_path = workdir / "agentlens.db"
    try:
        # 1-4. trace a scenario, detect (which now also persists), print the issue
        write_store = SQLiteTraceStore(db_path)
        write_lens = AgentLens(store=write_store)
        run_id = run_inefficient_agent(write_lens)
        generated = write_lens.detect(run_id)
        print(f"detect() generated + persisted {len(generated)} issue(s):")
        for issue in generated:
            print(f"  {_describe(issue)}")

        # 5-6. close and reopen with a fresh store / lens
        write_store.close()
        read_store = SQLiteTraceStore(db_path)
        read_lens = AgentLens(store=read_store)

        # 7-9. read persisted issues WITHOUT running detection again
        restored = read_lens.get_issues(run_id)
        print(f"\nget_issues() after reopen returned {len(restored)} issue(s):")
        for issue in restored:
            print(f"  {_describe(issue)}")
        print(f"\nrestored == generated (model equality): {restored == generated}")

        read_store.close()
    finally:
        # 10. clean up
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

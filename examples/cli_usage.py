"""Drive the ``agentlens`` CLI programmatically against a temporary database.

Fully offline and self-contained: creates a throwaway SQLite database, persists
one detected demo run, then invokes ``agentlens.cli.main`` for each command and
prints the output. The temp directory is removed on exit -- no repository
artifacts are left behind.

Usage::

    python -m examples.cli_usage
"""

from __future__ import annotations

import io
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from agentlens import AgentLens
from agentlens.cli import main
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import run_retrying_agent


def _run_cli(*argv: str) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv))
    print(f"$ agentlens {' '.join(argv)}   (exit {code})")
    return buffer.getvalue()


def main_example() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlens-cli-"))
    db_path = workdir / "agentlens.db"
    try:
        # 1-5. persist one detected run, then drop the store
        store = SQLiteTraceStore(db_path)
        lens = AgentLens(store=store)
        run_id = run_retrying_agent(lens)
        lens.detect(run_id)
        store.close()

        db = str(db_path)
        rid = str(run_id)

        print(_run_cli("runs", "--db", db))
        print(_run_cli("issues", rid, "--db", db))
        print("--- report (first 20 lines) ---")
        print("\n".join(_run_cli("report", rid, "--db", db).splitlines()[:20]))
        print("...\n")
        print("--- export json (first 8 lines) ---")
        print("\n".join(_run_cli("export", rid, "--format", "json", "--db", db).splitlines()[:8]))
        print("...\n")
        print("--- export markdown (first 10 lines) ---")
        print(
            "\n".join(_run_cli("export", rid, "--format", "markdown", "--db", db).splitlines()[:10])
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main_example()

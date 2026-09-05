"""Trace a scenario, detect, build a report, and export it as JSON and Markdown.

Fully offline and deterministic. Uses a throwaway temp-directory SQLite database
that is removed on exit -- no repository artifacts are left behind.

Usage::

    python -m examples.export_report
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import run_retrying_agent


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlens-export-"))
    db_path = workdir / "agentlens.db"
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        run_id = run_retrying_agent(lens)
        lens.detect(run_id)

        report = lens.get_report(run_id)
        json_output = export_json(report)
        markdown_output = export_markdown(report)

        print("=== JSON export (first 15 lines) ===")
        for line in json_output.splitlines()[:15]:
            print(line)
        print("...")
        print(f"\n(JSON export is {len(json_output)} characters)\n")

        print("=== Markdown export ===")
        print(markdown_output)
    finally:
        store.close()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

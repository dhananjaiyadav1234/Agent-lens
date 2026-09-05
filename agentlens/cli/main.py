"""``agentlens`` CLI: parser, entry point, and one thin helper per command.

Read-only. No command runs detectors, calls ``detect``, or writes to storage.
All database access goes through ``SQLiteTraceStore`` / ``AgentLens`` -- there is
no direct SQL and no hidden configuration (the database path is always explicit
via ``--db``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from agentlens import AgentLens, AgentLensError
from agentlens.export import export_json, export_markdown
from agentlens.models import AgentIssue, AgentRun
from agentlens.storage import SQLiteTraceStore

__all__ = ["build_parser", "main"]

_NOT_FINISHED = "Not finished"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``agentlens`` argument parser (``runs`` / ``issues`` / ``report`` / ``export``)."""

    parser = argparse.ArgumentParser(
        prog="agentlens",
        description="Inspect a persisted AgentLens SQLite database (read-only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    def _add_db(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--db",
            required=True,
            metavar="PATH",
            help="path to the SQLite database file",
        )

    runs_parser = subparsers.add_parser("runs", help="list every run in the database")
    _add_db(runs_parser)
    runs_parser.set_defaults(handler=_run_runs_command)

    issues_parser = subparsers.add_parser("issues", help="show persisted issues for one run")
    issues_parser.add_argument("run_id", metavar="RUN_ID", help="the run id (a UUID)")
    _add_db(issues_parser)
    issues_parser.set_defaults(handler=_run_issues_command)

    report_parser = subparsers.add_parser("report", help="print one run's Markdown report")
    report_parser.add_argument("run_id", metavar="RUN_ID", help="the run id (a UUID)")
    _add_db(report_parser)
    report_parser.set_defaults(handler=_run_report_command)

    export_parser = subparsers.add_parser(
        "export", help="export one run's report as JSON or Markdown"
    )
    export_parser.add_argument("run_id", metavar="RUN_ID", help="the run id (a UUID)")
    export_parser.add_argument(
        "--format",
        required=True,
        choices=("json", "markdown"),
        help="export format",
    )
    _add_db(export_parser)
    export_parser.set_defaults(handler=_run_export_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (or ``sys.argv[1:]``) and run the selected command.

    Returns an exit code: ``0`` on success, ``1`` on an expected user/application
    error (invalid run id, unknown run, unopenable database directory). Argument
    parsing errors exit via argparse's normal behaviour (status ``2``).
    """

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    return args.handler(args)


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------


def _run_runs_command(args: argparse.Namespace) -> int:
    if (message := _db_directory_error(args.db)) is not None:
        return _error(message)
    with SQLiteTraceStore(args.db) as store:
        runs = AgentLens(store=store).list_runs()
    _print_runs(runs)
    return 0


def _run_issues_command(args: argparse.Namespace) -> int:
    run_id = _parse_run_id(args.run_id)
    if run_id is None:
        return _error(f"error: invalid run id: {args.run_id}")
    if (message := _db_directory_error(args.db)) is not None:
        return _error(message)

    with SQLiteTraceStore(args.db) as store:
        lens = AgentLens(store=store)
        if lens.get_run(run_id) is None:
            return _error(f"error: no run found with id {run_id}")
        issues = lens.get_issues(run_id)

    if not issues:
        print("No issues found.")
        return 0
    _print_issues(issues)
    return 0


def _run_report_command(args: argparse.Namespace) -> int:
    run_id = _parse_run_id(args.run_id)
    if run_id is None:
        return _error(f"error: invalid run id: {args.run_id}")
    if (message := _db_directory_error(args.db)) is not None:
        return _error(message)

    with SQLiteTraceStore(args.db) as store:
        lens = AgentLens(store=store)
        try:
            report = lens.get_report(run_id)
        except AgentLensError:
            return _error(f"error: no run found with id {run_id}")

    sys.stdout.write(export_markdown(report))
    return 0


def _run_export_command(args: argparse.Namespace) -> int:
    run_id = _parse_run_id(args.run_id)
    if run_id is None:
        return _error(f"error: invalid run id: {args.run_id}")
    if (message := _db_directory_error(args.db)) is not None:
        return _error(message)

    with SQLiteTraceStore(args.db) as store:
        lens = AgentLens(store=store)
        try:
            report = lens.get_report(run_id)
        except AgentLensError:
            return _error(f"error: no run found with id {run_id}")

    rendered = export_json(report) if args.format == "json" else export_markdown(report)
    sys.stdout.write(rendered)
    return 0


# ---------------------------------------------------------------------------
# Rendering (plain text; JSON/Markdown come from the export layer verbatim)
# ---------------------------------------------------------------------------


def _print_runs(runs: list[AgentRun]) -> None:
    if not runs:
        print("No runs found.")
        return
    print(f"Runs: {len(runs)}")
    for position, run in enumerate(runs, start=1):
        finished = run.finished_at.isoformat() if run.finished_at is not None else _NOT_FINISHED
        print()
        print(f"{position}. {run.id}")
        print(f"   Task: {run.task}")
        print(f"   Status: {run.status.value}")
        print(f"   Started At: {run.started_at.isoformat()}")
        print(f"   Finished At: {finished}")


def _print_issues(issues: list[AgentIssue]) -> None:
    print(f"Issues: {len(issues)}")
    for position, issue in enumerate(issues, start=1):
        print()
        print(f"{position}. Issue ID: {issue.id}")
        print(f"   Type: {issue.issue_type.value}")
        print(f"   Severity: {issue.severity.value}")
        print(f"   Description: {issue.description}")
        if issue.related_event_ids:
            print("   Related Events:")
            for event_id in issue.related_event_ids:
                print(f"     - {event_id}")
        else:
            print("   Related Events: (none)")
        print("   Metadata:")
        for line in _metadata_json(issue.metadata).splitlines():
            print(f"     {line}")


def _metadata_json(metadata: dict) -> str:
    return json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_run_id(raw: str) -> UUID | None:
    try:
        return UUID(raw)
    except ValueError:
        return None


def _db_directory_error(db: str) -> str | None:
    parent = Path(db).parent
    if str(parent) not in ("", ".") and not parent.is_dir():
        return f"error: database directory does not exist: {parent}"
    return None


def _error(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

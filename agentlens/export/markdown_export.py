"""Deterministic Markdown export of an :class:`~agentlens.models.AgentReport`.

Section order is fixed: Title -> Run -> Summary -> Events -> Issues. Events,
issues, and every summary mapping are rendered in the exact order the report
holds them (never sorted). JSON-shaped fields (``input`` / ``output`` /
``metadata``) are rendered as fenced ``json`` blocks with sorted keys so the
output is byte-identical across repeated calls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

from agentlens.models import AgentEvent, AgentIssue, AgentReport
from agentlens.models.base import JsonValue

__all__ = ["export_markdown"]

_EMPTY = "_None._"
_NOT_FINISHED = "Not finished"
_NO_EVENTS = "_No events recorded._"
_NO_ISSUES = "_No issues detected._"


def export_markdown(report: AgentReport) -> str:
    """Render ``report`` as a deterministic human-readable Markdown document.

    Pure: ``report`` is not mutated. Repeated calls return byte-identical output;
    nothing derived from the clock, randomness, or object identity appears.
    """

    lines: list[str] = ["# AgentLens Report", ""]
    lines += _run_section(report)
    lines += _summary_section(report)
    lines += _events_section(report.events)
    lines += _issues_section(report.issues)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _run_section(report: AgentReport) -> list[str]:
    run = report.run
    finished = _fmt_dt(run.finished_at) if run.finished_at is not None else _NOT_FINISHED
    return [
        "## Run",
        "",
        f"- **Run ID:** `{run.id}`",
        f"- **Task:** {run.task}",
        f"- **Status:** {run.status.value}",
        f"- **Started At:** {_fmt_dt(run.started_at)}",
        f"- **Finished At:** {finished}",
        "",
    ]


def _summary_section(report: AgentReport) -> list[str]:
    summary = report.summary
    lines = [
        "## Summary",
        "",
        f"- **Total Events:** {summary.total_events}",
        f"- **Total Issues:** {summary.total_issues}",
        "",
        "### Events by Type",
        "",
        *_count_table("Event Type", summary.events_by_type),
        "",
        "### Issues by Type",
        "",
        *_count_table("Issue Type", summary.issues_by_type),
        "",
        "### Issues by Severity",
        "",
        *_count_table("Severity", summary.issues_by_severity),
        "",
    ]
    return lines


def _events_section(events: list[AgentEvent]) -> list[str]:
    lines = ["## Events", ""]
    if not events:
        lines += [_NO_EVENTS, ""]
        return lines
    for event in events:
        lines += _event_block(event)
    return lines


def _issues_section(issues: list[AgentIssue]) -> list[str]:
    lines = ["## Issues", ""]
    if not issues:
        lines += [_NO_ISSUES, ""]
        return lines
    for position, issue in enumerate(issues, start=1):
        lines += _issue_block(position, issue)
    return lines


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _event_block(event: AgentEvent) -> list[str]:
    duration = "None" if event.duration_ms is None else json.dumps(event.duration_ms)
    status = "None" if event.status is None else event.status
    return [
        f"### Event {event.sequence_number}",
        "",
        f"- **Type:** {event.event_type.value}",
        f"- **Name:** {event.name}",
        f"- **Timestamp:** {_fmt_dt(event.timestamp)}",
        f"- **Duration (ms):** {duration}",
        f"- **Status:** {status}",
        "- **Input:**",
        "",
        *_json_block(event.input),
        "",
        "- **Output:**",
        "",
        *_json_block(event.output),
        "",
        "- **Metadata:**",
        "",
        *_json_block(event.metadata),
        "",
    ]


def _issue_block(position: int, issue: AgentIssue) -> list[str]:
    lines = [
        f"### Issue {position}",
        "",
        f"- **Issue ID:** `{issue.id}`",
        f"- **Type:** {issue.issue_type.value}",
        f"- **Severity:** {issue.severity.value}",
        f"- **Description:** {issue.description}",
        "- **Related Events:**",
    ]
    if issue.related_event_ids:
        lines += [f"  - `{event_id}`" for event_id in issue.related_event_ids]
    else:
        lines.append(f"  {_EMPTY}")
    lines += [
        "",
        "#### Metadata",
        "",
        *_json_block(issue.metadata),
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_table(header: str, counts: Mapping[object, int]) -> list[str]:
    if not counts:
        return [_EMPTY]
    rows = [f"| {header} | Count |", "|---|---:|"]
    rows += [f"| {key.value} | {value} |" for key, value in counts.items()]
    return rows


def _json_block(value: JsonValue | None) -> list[str]:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    return ["```json", rendered, "```"]


def _fmt_dt(value: datetime) -> str:
    return value.isoformat()

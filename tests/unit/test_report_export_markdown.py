"""Unit tests for :func:`agentlens.export.export_markdown`."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agentlens.export import export_markdown
from agentlens.models import (
    AgentEvent,
    AgentIssue,
    AgentReport,
    AgentRun,
    EventType,
    IssueType,
    RunStatus,
    Severity,
)

_RUN_ID = uuid4()
_STARTED = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _run(**kw) -> AgentRun:
    base = {
        "id": _RUN_ID,
        "task": "look things up",
        "status": RunStatus.SUCCESS,
        "started_at": _STARTED,
        "finished_at": _STARTED + timedelta(seconds=3),
        "metadata": {},
    }
    base.update(kw)
    return AgentRun(**base)


def _event(seq, event_type=EventType.DECISION, name="decide", **kw) -> AgentEvent:
    return AgentEvent(run_id=_RUN_ID, sequence_number=seq, event_type=event_type, name=name, **kw)


def _issue(**kw) -> AgentIssue:
    base = {
        "run_id": _RUN_ID,
        "issue_type": IssueType.AGENT_LOOP,
        "severity": Severity.MEDIUM,
        "description": "loop",
    }
    base.update(kw)
    return AgentIssue(**base)


def _report(events=None, issues=None, run=None) -> AgentReport:
    return AgentReport.build(run or _run(), events or [], issues or [])


def test_required_sections_and_title_order():
    md = export_markdown(_report([_event(0)], [_issue()]))
    assert md.startswith("# AgentLens Report\n")
    assert (
        md.index("## Run") < md.index("## Summary") < md.index("## Events") < md.index("## Issues")
    )
    for heading in ["### Events by Type", "### Issues by Type", "### Issues by Severity"]:
        assert heading in md


def test_run_fields_present():
    md = export_markdown(_report([_event(0)]))
    assert f"- **Run ID:** `{_RUN_ID}`" in md
    assert "- **Task:** look things up" in md
    assert "- **Status:** success" in md
    assert f"- **Started At:** {_STARTED.isoformat()}" in md
    assert f"- **Finished At:** {(_STARTED + timedelta(seconds=3)).isoformat()}" in md


def test_finished_at_none_renders_not_finished():
    running = AgentRun(id=_RUN_ID, task="t")
    md = export_markdown(AgentReport.build(running, [_event(0)], []))
    assert "- **Finished At:** Not finished" in md


def test_summary_counts_and_first_appearance_order_preserved():
    events = [
        _event(0, EventType.RUN_STARTED, "run_started"),
        _event(1, EventType.DECISION),
        _event(2, EventType.TOOL_CALL_STARTED, "t"),
        _event(3, EventType.DECISION),
        _event(4, EventType.RUN_COMPLETED, "run_completed"),
    ]
    issues = [
        _issue(issue_type=IssueType.EXCESSIVE_RETRY, severity=Severity.HIGH),
        _issue(issue_type=IssueType.AGENT_LOOP, severity=Severity.MEDIUM),
        _issue(issue_type=IssueType.EXCESSIVE_RETRY, severity=Severity.HIGH),
    ]
    md = export_markdown(_report(events, issues))

    assert "- **Total Events:** 5" in md
    assert "- **Total Issues:** 3" in md

    events_table = md.split("### Events by Type")[1].split("###")[0]
    assert events_table.index("run_started") < events_table.index("decision")
    assert events_table.index("decision") < events_table.index("tool_call_started")
    assert "| decision | 2 |" in events_table

    types_table = md.split("### Issues by Type")[1].split("###")[0]
    assert types_table.index("excessive_retry") < types_table.index("agent_loop")
    assert "| excessive_retry | 2 |" in types_table

    sev_table = md.split("### Issues by Severity")[1].split("##")[0]
    assert "| high | 2 |" in sev_table
    assert "| medium | 1 |" in sev_table
    assert sev_table.index("high") < sev_table.index("medium")


def test_empty_summary_mappings_render_none_placeholder():
    md = export_markdown(_report([_event(0)], []))  # no issues
    issues_by_type = md.split("### Issues by Type")[1].split("###")[0]
    issues_by_sev = md.split("### Issues by Severity")[1].split("##")[0]
    assert "_None._" in issues_by_type
    assert "_None._" in issues_by_sev


def test_event_order_is_preserved_verbatim():
    events = [_event(5, name="a"), _event(2, name="b"), _event(9, name="c")]
    md = export_markdown(_report(events))
    assert md.index("### Event 5") < md.index("### Event 2") < md.index("### Event 9")


def test_issue_order_is_preserved_verbatim():
    issues = [
        _issue(description="first", severity=Severity.CRITICAL),
        _issue(description="second", severity=Severity.MEDIUM),
        _issue(description="third", severity=Severity.HIGH),
    ]
    md = export_markdown(_report([_event(0)], issues))
    assert md.index("first") < md.index("second") < md.index("third")
    assert "### Issue 1" in md and "### Issue 2" in md and "### Issue 3" in md


def test_no_issues_renders_explicit_state():
    md = export_markdown(_report([_event(0)], []))
    assert "## Issues\n\n_No issues detected._" in md


def test_event_json_fields_are_valid_json_blocks():
    payload = {"q": "café", "nested": {"list": [1, 2.5, True, None]}, "flag": False}
    md = export_markdown(_report([_event(0, input=payload, output=["a", None, 3], metadata={})]))
    blocks = [b.split("```")[0] for b in md.split("```json\n")[1:]]
    parsed = [json.loads(b) for b in blocks]
    assert payload in parsed
    assert ["a", None, 3] in parsed
    assert {} in parsed


def test_none_scalar_fields_render_deterministically():
    md = export_markdown(
        _report([_event(0, duration_ms=None, status=None, input=None, output=None)])
    )
    assert "- **Duration (ms):** None" in md
    assert "- **Status:** None" in md
    # None input/output become the json literal null
    assert md.count("```json\nnull\n```") >= 2


def test_related_event_ids_rendered_and_empty_state():
    e1, e2 = uuid4(), uuid4()
    md = export_markdown(
        _report(
            [_event(0)],
            [_issue(related_event_ids=[e1, e2]), _issue(description="lonely")],
        )
    )
    assert f"  - `{e1}`" in md
    assert f"  - `{e2}`" in md
    # the second issue has no related events
    lonely = md.split("lonely")[1]
    assert "_None._" in lonely.split("#### Metadata")[0]


def test_repeated_calls_are_byte_identical():
    report = _report(
        [_event(0, input={"x": 1}), _event(1, event_type=EventType.RUN_COMPLETED, name="rc")],
        [_issue(), _issue(severity=Severity.HIGH, issue_type=IssueType.INEFFICIENCY)],
    )
    assert export_markdown(report) == export_markdown(report) == export_markdown(report)


def test_does_not_mutate_report():
    report = _report(
        [_event(0, input={"nested": {"a": [1]}}, metadata={"z": 1, "a": 2})],
        [_issue(metadata={"detector": "x"})],
    )
    before = report.model_dump(mode="json")
    export_markdown(report)
    export_markdown(report)
    assert report.model_dump(mode="json") == before


def test_unicode_preserved_in_markdown():
    md = export_markdown(_report([_event(0, name="日本語 café 🚀")], run=_run(task="τ task")))
    assert "日本語 café 🚀" in md
    assert "τ task" in md

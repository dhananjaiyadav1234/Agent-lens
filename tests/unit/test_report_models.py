"""Unit tests for the report models (:class:`ReportSummary`, :class:`AgentReport`)."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agentlens.models import (
    AgentEvent,
    AgentIssue,
    AgentReport,
    AgentRun,
    EventType,
    IssueType,
    ReportSummary,
    RunStatus,
    Severity,
)

_RUN_ID = uuid4()
_STARTED = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _run(**overrides) -> AgentRun:
    base = {"id": _RUN_ID, "task": "t"}
    base.update(overrides)
    return AgentRun(**base)


def _event(seq, event_type, name="step") -> AgentEvent:
    return AgentEvent(run_id=_RUN_ID, sequence_number=seq, event_type=event_type, name=name)


def _issue(issue_type=IssueType.AGENT_LOOP, severity=Severity.MEDIUM) -> AgentIssue:
    return AgentIssue(run_id=_RUN_ID, issue_type=issue_type, severity=severity, description="d")


# ---------------------------------------------------------------------------
# ReportSummary
# ---------------------------------------------------------------------------


def test_summary_construction_and_totals():
    events = [
        _event(0, EventType.RUN_STARTED),
        _event(1, EventType.DECISION),
        _event(2, EventType.TOOL_CALL_STARTED),
        _event(3, EventType.TOOL_CALL_COMPLETED),
        _event(4, EventType.RUN_COMPLETED),
    ]
    summary = ReportSummary.from_events_and_issues(events, [])
    assert summary.total_events == 5
    assert summary.total_issues == 0
    assert summary.issues_by_type == {}
    assert summary.issues_by_severity == {}


def test_events_by_type_counts_and_first_appearance_order():
    events = [
        _event(0, EventType.RUN_STARTED),
        _event(1, EventType.DECISION),
        _event(2, EventType.DECISION),
        _event(3, EventType.TOOL_CALL_STARTED),
        _event(4, EventType.DECISION),
        _event(5, EventType.TOOL_CALL_STARTED),
        _event(6, EventType.RUN_COMPLETED),
    ]
    summary = ReportSummary.from_events_and_issues(events, [])
    assert summary.events_by_type == {
        EventType.RUN_STARTED: 1,
        EventType.DECISION: 3,
        EventType.TOOL_CALL_STARTED: 2,
        EventType.RUN_COMPLETED: 1,
    }
    assert list(summary.events_by_type) == [
        EventType.RUN_STARTED,
        EventType.DECISION,
        EventType.TOOL_CALL_STARTED,
        EventType.RUN_COMPLETED,
    ]
    # zero-count enum members are not added
    assert EventType.ERROR not in summary.events_by_type


def test_issue_counts_and_first_appearance_order_with_repeats():
    issues = [
        _issue(IssueType.EXCESSIVE_RETRY, Severity.HIGH),
        _issue(IssueType.AGENT_LOOP, Severity.MEDIUM),
        _issue(IssueType.EXCESSIVE_RETRY, Severity.HIGH),
        _issue(IssueType.INEFFICIENCY, Severity.CRITICAL),
        _issue(IssueType.AGENT_LOOP, Severity.MEDIUM),
    ]
    summary = ReportSummary.from_events_and_issues([], issues)
    assert summary.total_issues == 5
    assert summary.issues_by_type == {
        IssueType.EXCESSIVE_RETRY: 2,
        IssueType.AGENT_LOOP: 2,
        IssueType.INEFFICIENCY: 1,
    }
    assert list(summary.issues_by_type) == [
        IssueType.EXCESSIVE_RETRY,
        IssueType.AGENT_LOOP,
        IssueType.INEFFICIENCY,
    ]
    assert summary.issues_by_severity == {
        Severity.HIGH: 2,
        Severity.MEDIUM: 2,
        Severity.CRITICAL: 1,
    }
    assert list(summary.issues_by_severity) == [Severity.HIGH, Severity.MEDIUM, Severity.CRITICAL]


def test_summary_empty_inputs():
    summary = ReportSummary.from_events_and_issues([], [])
    assert summary.total_events == 0
    assert summary.total_issues == 0
    assert summary.events_by_type == {}
    assert summary.issues_by_type == {}
    assert summary.issues_by_severity == {}


def test_summary_json_serialization_and_round_trip():
    events = [_event(0, EventType.RUN_STARTED), _event(1, EventType.DECISION)]
    issues = [_issue(IssueType.AGENT_LOOP, Severity.MEDIUM)]
    summary = ReportSummary.from_events_and_issues(events, issues)

    dumped = summary.model_dump(mode="json")
    assert dumped["events_by_type"] == {"run_started": 1, "decision": 1}
    assert dumped["issues_by_type"] == {"agent_loop": 1}
    assert dumped["issues_by_severity"] == {"medium": 1}

    restored = ReportSummary.model_validate_json(summary.model_dump_json())
    assert restored == summary
    assert all(isinstance(k, EventType) for k in restored.events_by_type)
    assert all(isinstance(k, IssueType) for k in restored.issues_by_type)
    assert all(isinstance(k, Severity) for k in restored.issues_by_severity)


# ---------------------------------------------------------------------------
# AgentReport
# ---------------------------------------------------------------------------


def _completed_run() -> AgentRun:
    return _run(
        status=RunStatus.SUCCESS,
        started_at=_STARTED,
        finished_at=_STARTED + timedelta(seconds=1),
    )


def test_report_build_preserves_run_events_issues_verbatim():
    run = _completed_run()
    events = [_event(0, EventType.RUN_STARTED), _event(1, EventType.RUN_COMPLETED)]
    issues = [_issue()]
    report = AgentReport.build(run, events, issues)

    assert report.run == run
    assert report.events == events
    assert report.issues == issues
    assert report.summary.total_events == 2
    assert report.summary.total_issues == 1


def test_report_build_does_not_reorder():
    run = _completed_run()
    events = [
        _event(2, EventType.DECISION),
        _event(5, EventType.ERROR),
        _event(9, EventType.DECISION),
    ]
    report = AgentReport.build(run, events, [])
    assert [e.sequence_number for e in report.events] == [2, 5, 9]  # exactly as given


def test_report_json_round_trip():
    run = _completed_run()
    events = [
        _event(0, EventType.RUN_STARTED),
        _event(1, EventType.TOOL_CALL_STARTED),
        _event(2, EventType.TOOL_CALL_COMPLETED),
        _event(3, EventType.RUN_COMPLETED),
    ]
    issues = [
        _issue(IssueType.DUPLICATE_TOOL_CALL, Severity.MEDIUM),
        _issue(IssueType.DUPLICATE_TOOL_CALL, Severity.HIGH),
    ]
    report = AgentReport.build(run, events, issues)

    restored = AgentReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.model_dump() == report.model_dump()
    assert restored.run == run
    assert restored.events == events
    assert restored.issues == issues
    assert restored.summary == report.summary


def test_report_model_dump_json_mode_is_pure_json():
    run = _completed_run()
    report = AgentReport.build(run, [_event(0, EventType.RUN_STARTED)], [_issue()])
    dumped = report.model_dump(mode="json")
    assert isinstance(dumped["run"]["id"], str)
    assert isinstance(dumped["events"], list)
    assert dumped["summary"]["events_by_type"] == {"run_started": 1}

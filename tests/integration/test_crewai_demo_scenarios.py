"""Integration: CrewAI event-bus activity through the full AgentLens pipeline.

Driven offline through CrewAI's real event bus (``crewai_event_bus.emit`` with
the SDK's own event classes). No crew, no models, no API keys, no network.
"""

import datetime as dt
import json

import crewai.events.event_listener as _crewai_event_listener
import pytest
from crewai.events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.integrations.crewai import AgentLensCrewAIListener
from agentlens.models import AgentReport, EventType, IssueType, RunStatus
from agentlens.storage import SQLiteTraceStore

_NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def _quiet_crewai_console():
    formatter = _crewai_event_listener.event_listener.formatter
    previous = formatter.verbose
    formatter.verbose = False
    yield
    formatter.verbose = previous


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _llm(model: str, answer: str) -> None:
    crewai_event_bus.emit("test", LLMCallStartedEvent(call_id="c", model=model))
    crewai_event_bus.emit(
        "test",
        LLMCallCompletedEvent(call_id="c", model=model, response=answer, call_type="llm_call"),
    )


def _tool(name: str, args: dict, *, output=None, error=None) -> None:
    crewai_event_bus.emit("test", ToolUsageStartedEvent(tool_name=name, tool_args=args))
    if error is not None:
        crewai_event_bus.emit(
            "test", ToolUsageErrorEvent(tool_name=name, tool_args=args, error=error)
        )
    else:
        crewai_event_bus.emit(
            "test",
            ToolUsageFinishedEvent(
                tool_name=name, tool_args=args, output=output, started_at=_NOW, finished_at=_NOW
            ),
        )


# ---------------------------------------------------------------------------
# Scenario A -- normal execution
# ---------------------------------------------------------------------------


def test_normal_execution():
    lens = AgentLens()
    with lens.trace("answer a support question") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _llm("planner", "call lookup_customer")
            _tool("lookup_customer", {"customer_id": "123"}, output={"open_orders": 2})

    run_id = trace.run_id
    issues = lens.detect(run_id)
    report = lens.get_report(run_id)

    assert lens.get_run(run_id).status is RunStatus.SUCCESS
    assert issues == []
    assert report.summary.total_events == len(report.events)
    assert report.summary.events_by_type[EventType.LLM_CALL_STARTED] == 1
    assert report.summary.events_by_type[EventType.LLM_CALL_COMPLETED] == 1
    assert report.summary.events_by_type[EventType.TOOL_CALL_STARTED] == 1
    assert report.summary.events_by_type[EventType.TOOL_CALL_COMPLETED] == 1
    seqs = [e.sequence_number for e in report.events]
    assert seqs == sorted(seqs)


# ---------------------------------------------------------------------------
# Scenario B -- retry pattern (existing RetryDetector, unchanged)
# ---------------------------------------------------------------------------


def test_retry_pattern_detected_by_existing_detector():
    lens = AgentLens()
    with lens.trace("fetch with retries") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            for _ in range(3):
                _tool("flaky_fetch", {"customer_id": "9"}, error="upstream failure")
            _tool("flaky_fetch", {"customer_id": "9"}, output={"id": "9"})

    issues = lens.detect(trace.run_id)
    retry_issues = [i for i in issues if i.issue_type is IssueType.EXCESSIVE_RETRY]
    assert len(retry_issues) == 1
    assert retry_issues[0].metadata["operation"] == "flaky_fetch"
    assert retry_issues[0].metadata["failed_attempts"] == 3


# ---------------------------------------------------------------------------
# Scenario C -- duplicate tool call
# ---------------------------------------------------------------------------


def test_duplicate_tool_call_detected_by_existing_detector():
    lens = AgentLens()
    with lens.trace("look up twice") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _tool("lookup_customer", {"customer_id": "123"}, output={"open_orders": 2})
            _tool("lookup_customer", {"customer_id": "123"}, output={"open_orders": 2})

    issues = lens.detect(trace.run_id)
    dup = [i for i in issues if i.issue_type is IssueType.DUPLICATE_TOOL_CALL]
    assert len(dup) == 1
    assert dup[0].metadata["operation"] == "lookup_customer"


# ---------------------------------------------------------------------------
# Scenario D -- error-heavy execution, persisted
# ---------------------------------------------------------------------------


def test_error_heavy_execution_records_and_persists(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        with lens.trace("error heavy") as trace:
            with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
                for i in range(3):
                    _tool("always_fails", {"x": i}, error="ValueError: bad")
        run_id = trace.run_id
        errors = [e for e in lens.get_events(run_id) if e.event_type is EventType.ERROR]
        assert len(errors) == 3
        assert all(e.output["operation"] == "always_fails" for e in errors)
        report = lens.get_report(run_id)
        assert report.summary.events_by_type[EventType.ERROR] == 3
    finally:
        store.close()

    store2 = SQLiteTraceStore(db_path)
    try:
        reopened = AgentLens(store=store2)
        assert len([e for e in reopened.get_events(run_id) if e.event_type is EventType.ERROR]) == 3
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# Scenario E -- SQLite close / reopen
# ---------------------------------------------------------------------------


def test_full_pipeline_survives_sqlite_close_and_reopen(db_path):
    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    with lens_a.trace("persisted crewai run") as trace:
        with AgentLensCrewAIListener(lens=lens_a, run_id=trace.run_id):
            _tool("lookup_customer", {"customer_id": "1"}, output={"open_orders": 1})
            _tool("lookup_customer", {"customer_id": "1"}, output={"open_orders": 1})
    run_id = trace.run_id
    issues_a = lens_a.detect(run_id)
    report_a = lens_a.get_report(run_id)
    md_a = export_markdown(report_a)
    store_a.close()
    del lens_a, store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        lens_b = AgentLens(store=store_b)
        assert [e.model_dump(mode="json") for e in lens_b.get_events(run_id)] == [
            e.model_dump(mode="json") for e in report_a.events
        ]
        assert [i.model_dump(mode="json") for i in lens_b.get_issues(run_id)] == [
            i.model_dump(mode="json") for i in issues_a
        ]
        assert export_markdown(lens_b.get_report(run_id)) == md_a
        assert len(lens_b.get_issues(run_id)) == len(issues_a)
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# Scenario F -- export
# ---------------------------------------------------------------------------


def test_export_layer_works_unchanged():
    lens = AgentLens()
    with lens.trace("export me") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _llm("m", "ok")
            _tool("lookup_customer", {"customer_id": "42"}, output={"open_orders": 0})
    lens.detect(trace.run_id)
    report = lens.get_report(trace.run_id)

    parsed = json.loads(export_json(report))
    assert parsed == report.model_dump(mode="json")
    assert AgentReport.model_validate(parsed) == report

    markdown_output = export_markdown(report)
    assert "# AgentLens Report" in markdown_output
    assert "lookup_customer" in markdown_output
    assert "crewai" in markdown_output


# ---------------------------------------------------------------------------
# Scenario G -- cross-run isolation
# ---------------------------------------------------------------------------


def test_cross_run_isolation_with_detection(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)

        with lens.trace("run a - duplicate") as ta:
            with AgentLensCrewAIListener(lens=lens, run_id=ta.run_id):
                _tool("lookup_customer", {"customer_id": "a"}, output={"x": 1})
                _tool("lookup_customer", {"customer_id": "a"}, output={"x": 1})

        with lens.trace("run b - clean") as tb:
            with AgentLensCrewAIListener(lens=lens, run_id=tb.run_id):
                _tool("lookup_customer", {"customer_id": "b"}, output={"x": 1})

        issues_a = lens.detect(ta.run_id)
        issues_b = lens.detect(tb.run_id)

        assert [i.issue_type for i in issues_a] == [IssueType.DUPLICATE_TOOL_CALL]
        assert issues_b == []

        a_event_ids = {e.id for e in lens.get_events(ta.run_id)}
        b_event_ids = {e.id for e in lens.get_events(tb.run_id)}
        assert a_event_ids.isdisjoint(b_event_ids)
        for issue in issues_a:
            assert issue.run_id == ta.run_id
            assert set(issue.related_event_ids) <= a_event_ids

        # exports of one run never mention the other's data
        assert "customer_id" in export_json(lens.get_report(ta.run_id))
        assert '"b"' not in export_json(lens.get_report(ta.run_id))
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CLI compatibility
# ---------------------------------------------------------------------------


def test_cli_reads_persisted_crewai_run(db_path, capsys):
    store = SQLiteTraceStore(db_path)
    lens = AgentLens(store=store)
    with lens.trace("cli run") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _tool("lookup_customer", {"customer_id": "7"}, output={"open_orders": 3})
    run_id = trace.run_id
    lens.detect(run_id)
    store.close()
    del lens, store

    from agentlens.cli import main

    db = str(db_path)
    assert main(["runs", "--db", db]) == 0
    assert str(run_id) in capsys.readouterr().out
    assert main(["report", str(run_id), "--db", db]) == 0
    report_out = capsys.readouterr().out
    assert "# AgentLens Report" in report_out
    assert "lookup_customer" in report_out
    assert main(["export", str(run_id), "--format", "json", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out)

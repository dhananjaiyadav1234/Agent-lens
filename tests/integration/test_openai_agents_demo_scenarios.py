"""Integration: OpenAI Agents SDK span activity through the full AgentLens pipeline.

Driven offline through the SDK's real tracing machinery (``agents.tracing.trace``
+ span factories + ``set_trace_processors``). No models, no API keys, no network.
"""

import json

import pytest
from agents.tracing import function_span, generation_span, set_trace_processors
from agents.tracing import trace as agents_trace

from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.integrations.openai_agents import AgentLensOpenAITracer
from agentlens.models import AgentReport, EventType, IssueType, RunStatus
from agentlens.storage import SQLiteTraceStore


@pytest.fixture(autouse=True)
def _reset_trace_processors():
    yield
    set_trace_processors([])


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _tool(name: str, args: dict, *, output=None, error=None):
    with function_span(name=name, input=json.dumps(args)) as span:
        if error is not None:
            span.set_error(error)
        else:
            span.span_data.output = output


def _run_activity(lens: AgentLens, run_id, body) -> None:
    tracer = AgentLensOpenAITracer(lens=lens, run_id=run_id)
    set_trace_processors([tracer])
    with agents_trace("workflow"):
        body()
    set_trace_processors([])


# ---------------------------------------------------------------------------
# Scenario A — normal execution
# ---------------------------------------------------------------------------


def test_normal_execution():
    lens = AgentLens()
    with lens.trace("answer a support question") as trace:

        def body():
            with generation_span(model="planner") as span:
                span.span_data.output = [{"role": "assistant", "content": "call lookup_customer"}]
            _tool("lookup_customer", {"customer_id": "123"}, output={"open_orders": 2})

        _run_activity(lens, trace.run_id, body)

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
    # ordering preserved
    seqs = [e.sequence_number for e in report.events]
    assert seqs == sorted(seqs)


# ---------------------------------------------------------------------------
# Scenario B — retry pattern (existing RetryDetector, unchanged)
# ---------------------------------------------------------------------------


def test_retry_pattern_detected_by_existing_detector():
    lens = AgentLens()
    with lens.trace("fetch with retries") as trace:

        def body():
            for _ in range(3):
                _tool(
                    "flaky_fetch",
                    {"customer_id": "9"},
                    error={
                        "message": "upstream failure",
                        "data": {"exception_type": "RuntimeError"},
                    },
                )
            _tool("flaky_fetch", {"customer_id": "9"}, output={"id": "9"})

        _run_activity(lens, trace.run_id, body)

    issues = lens.detect(trace.run_id)
    retry_issues = [i for i in issues if i.issue_type is IssueType.EXCESSIVE_RETRY]
    assert len(retry_issues) == 1
    assert retry_issues[0].metadata["operation"] == "flaky_fetch"
    assert retry_issues[0].metadata["failed_attempts"] == 3


# ---------------------------------------------------------------------------
# Scenario C — duplicate tool call
# ---------------------------------------------------------------------------


def test_duplicate_tool_call_detected_by_existing_detector():
    lens = AgentLens()
    with lens.trace("look up twice") as trace:

        def body():
            _tool("lookup_customer", {"customer_id": "123"}, output={"open_orders": 2})
            _tool("lookup_customer", {"customer_id": "123"}, output={"open_orders": 2})

        _run_activity(lens, trace.run_id, body)

    issues = lens.detect(trace.run_id)
    dup = [i for i in issues if i.issue_type is IssueType.DUPLICATE_TOOL_CALL]
    assert len(dup) == 1
    assert dup[0].metadata["operation"] == "lookup_customer"


# ---------------------------------------------------------------------------
# Scenario D — error-heavy execution
# ---------------------------------------------------------------------------


def test_error_heavy_execution_records_and_persists(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        with lens.trace("error heavy") as trace:

            def body():
                for i in range(3):
                    _tool(
                        "always_fails",
                        {"x": i},
                        error={"message": f"bad {i}", "data": {"exception_type": "ValueError"}},
                    )

            _run_activity(lens, trace.run_id, body)
        run_id = trace.run_id
        errors = [e for e in lens.get_events(run_id) if e.event_type is EventType.ERROR]
        assert len(errors) == 3
        assert all(e.output["exception_type"] == "ValueError" for e in errors)
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
# Scenario E — SQLite reopen
# ---------------------------------------------------------------------------


def test_full_pipeline_survives_sqlite_close_and_reopen(db_path):
    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    with lens_a.trace("persisted openai run") as trace:

        def body():
            _tool("lookup_customer", {"customer_id": "1"}, output={"open_orders": 1})
            _tool("lookup_customer", {"customer_id": "1"}, output={"open_orders": 1})

        _run_activity(lens_a, trace.run_id, body)
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
        # reporting/reopen created nothing
        assert len(lens_b.get_issues(run_id)) == len(issues_a)
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# Scenario F — export
# ---------------------------------------------------------------------------


def test_export_layer_works_unchanged():
    lens = AgentLens()
    with lens.trace("export me") as trace:

        def body():
            with generation_span(model="m") as span:
                span.span_data.output = [{"role": "assistant", "content": "ok"}]
            _tool("lookup_customer", {"customer_id": "42"}, output={"open_orders": 0})

        _run_activity(lens, trace.run_id, body)
    lens.detect(trace.run_id)
    report = lens.get_report(trace.run_id)

    json_output = export_json(report)
    parsed = json.loads(json_output)
    assert parsed == report.model_dump(mode="json")
    assert AgentReport.model_validate(parsed) == report

    markdown_output = export_markdown(report)
    assert "# AgentLens Report" in markdown_output
    assert "lookup_customer" in markdown_output
    assert "openai_agents" in markdown_output


# ---------------------------------------------------------------------------
# Scenario G — cross-run isolation
# ---------------------------------------------------------------------------


def test_cross_run_isolation_with_detection(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)

        with lens.trace("run a - duplicate") as ta:
            ta_tracer = AgentLensOpenAITracer(lens=lens, run_id=ta.run_id)
            set_trace_processors([ta_tracer])
            with agents_trace("wf a"):
                _tool("lookup_customer", {"customer_id": "a"}, output={"x": 1})
                _tool("lookup_customer", {"customer_id": "a"}, output={"x": 1})

        with lens.trace("run b - clean") as tb:
            tb_tracer = AgentLensOpenAITracer(lens=lens, run_id=tb.run_id)
            set_trace_processors([ta_tracer, tb_tracer])
            with agents_trace("wf b"):
                _tool("lookup_customer", {"customer_id": "b"}, output={"x": 1})
        set_trace_processors([])

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
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CLI compatibility
# ---------------------------------------------------------------------------


def test_cli_reads_persisted_openai_run(db_path, capsys):
    store = SQLiteTraceStore(db_path)
    lens = AgentLens(store=store)
    with lens.trace("cli run") as trace:

        def body():
            _tool("lookup_customer", {"customer_id": "7"}, output={"open_orders": 3})

        _run_activity(lens, trace.run_id, body)
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

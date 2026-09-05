"""Integration: LangChain callback activity flowing through the full AgentLens pipeline.

Uses real local ``langchain_core`` tools invoked offline (no LLM, no network),
then the existing detect / report / export / SQLite layers unchanged.
"""

import json

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.tools import tool

from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.integrations.langchain import AgentLensCallbackHandler
from agentlens.models import EventType, IssueType, RunStatus
from agentlens.storage import SQLiteTraceStore


@tool
def lookup_customer(customer_id: str) -> dict:
    """Deterministic offline customer lookup."""

    return {"id": customer_id, "open_orders": 2}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _handler(lens, run_id):
    return AgentLensCallbackHandler(lens=lens, run_id=run_id)


# ---------------------------------------------------------------------------
# Normal execution
# ---------------------------------------------------------------------------


def test_normal_execution_records_events_and_has_no_issues():
    lens = AgentLens()
    with lens.trace("answer a support question") as trace:
        handler = _handler(lens, trace.run_id)
        from uuid import uuid4

        rid = uuid4()
        handler.on_llm_start({"name": "planner"}, ["What tool?"], run_id=rid)
        handler.on_llm_end(
            LLMResult(generations=[[Generation(text="use lookup_customer")]]), run_id=rid
        )
        lookup_customer.invoke({"customer_id": "123"}, config={"callbacks": [handler]})

    run_id = trace.run_id
    issues = lens.detect(run_id)
    report = lens.get_report(run_id)

    assert lens.get_run(run_id).status is RunStatus.SUCCESS
    assert issues == []
    assert report.summary.total_events == len(report.events)
    assert report.summary.events_by_type[EventType.LLM_CALL_STARTED] == 1
    assert report.summary.events_by_type[EventType.TOOL_CALL_STARTED] == 1


# ---------------------------------------------------------------------------
# Retry pattern (existing RetryDetector, unchanged)
# ---------------------------------------------------------------------------


def test_retry_pattern_is_detected_by_existing_detector():
    attempts = {"n": 0}

    @tool
    def flaky_fetch(customer_id: str) -> dict:
        """Fails a few times, then succeeds."""

        attempts["n"] += 1
        if attempts["n"] <= 3:
            raise RuntimeError("temporary upstream failure")
        return {"id": customer_id}

    lens = AgentLens()
    with lens.trace("fetch with retries") as trace:
        handler = _handler(lens, trace.run_id)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                flaky_fetch.invoke({"customer_id": "9"}, config={"callbacks": [handler]})
        flaky_fetch.invoke({"customer_id": "9"}, config={"callbacks": [handler]})

    issues = lens.detect(trace.run_id)
    retry_issues = [i for i in issues if i.issue_type is IssueType.EXCESSIVE_RETRY]
    assert len(retry_issues) == 1
    assert retry_issues[0].metadata["operation"] == "flaky_fetch"
    assert retry_issues[0].metadata["failed_attempts"] == 3


# ---------------------------------------------------------------------------
# Repeated identical tool call (existing DuplicateToolDetector, unchanged)
# ---------------------------------------------------------------------------


def test_repeated_identical_tool_call_is_detected_as_duplicate():
    lens = AgentLens()
    with lens.trace("look up twice") as trace:
        handler = _handler(lens, trace.run_id)
        lookup_customer.invoke({"customer_id": "123"}, config={"callbacks": [handler]})
        lookup_customer.invoke({"customer_id": "123"}, config={"callbacks": [handler]})

    issues = lens.detect(trace.run_id)
    dup = [i for i in issues if i.issue_type is IssueType.DUPLICATE_TOOL_CALL]
    assert len(dup) == 1
    assert dup[0].metadata["operation"] == "lookup_customer"


# ---------------------------------------------------------------------------
# Error-heavy execution
# ---------------------------------------------------------------------------


def test_errors_are_recorded_and_persisted(db_path):
    @tool
    def always_fails(x: int) -> str:
        """Always raises."""
        raise ValueError(f"bad value {x}")

    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        with lens.trace("error heavy") as trace:
            handler = _handler(lens, trace.run_id)
            for i in range(3):
                with pytest.raises(ValueError):
                    always_fails.invoke({"x": i}, config={"callbacks": [handler]})
        run_id = trace.run_id
        errors = [e for e in lens.get_events(run_id) if e.event_type is EventType.ERROR]
        assert len(errors) == 3
        assert all(e.output["exception_type"] == "ValueError" for e in errors)
    finally:
        store.close()

    # persisted
    store2 = SQLiteTraceStore(db_path)
    try:
        reopened = AgentLens(store=store2)
        assert len([e for e in reopened.get_events(run_id) if e.event_type is EventType.ERROR]) == 3
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# SQLite reopen
# ---------------------------------------------------------------------------


def test_full_pipeline_survives_sqlite_close_and_reopen(db_path):
    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    with lens_a.trace("persisted langchain run") as trace:
        handler = _handler(lens_a, trace.run_id)
        lookup_customer.invoke({"customer_id": "1"}, config={"callbacks": [handler]})
        lookup_customer.invoke({"customer_id": "1"}, config={"callbacks": [handler]})
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
        report_b = lens_b.get_report(run_id)
        assert export_markdown(report_b) == md_a
        # detecting again after reopen appends a further batch (existing semantics)
        again = lens_b.detect(run_id)
        assert len(lens_b.get_issues(run_id)) == len(issues_a) + len(again)
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# Export compatibility
# ---------------------------------------------------------------------------


def test_export_layer_works_unchanged_on_langchain_events():
    lens = AgentLens()
    with lens.trace("export me") as trace:
        handler = _handler(lens, trace.run_id)
        lookup_customer.invoke({"customer_id": "42"}, config={"callbacks": [handler]})
    lens.detect(trace.run_id)
    report = lens.get_report(trace.run_id)

    json_output = export_json(report)
    assert json.loads(json_output) == report.model_dump(mode="json")

    markdown_output = export_markdown(report)
    assert "# AgentLens Report" in markdown_output
    assert "lookup_customer" in markdown_output
    assert "langchain" in markdown_output  # framework marker in event metadata


# ---------------------------------------------------------------------------
# Cross-run isolation with detection
# ---------------------------------------------------------------------------


def test_cross_run_isolation_with_detection(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)

        with lens.trace("run a - duplicate") as ta:
            ha = _handler(lens, ta.run_id)
            lookup_customer.invoke({"customer_id": "a"}, config={"callbacks": [ha]})
            lookup_customer.invoke({"customer_id": "a"}, config={"callbacks": [ha]})

        with lens.trace("run b - clean") as tb:
            hb = _handler(lens, tb.run_id)
            lookup_customer.invoke({"customer_id": "b"}, config={"callbacks": [hb]})

        issues_a = lens.detect(ta.run_id)
        issues_b = lens.detect(tb.run_id)

        assert [i.issue_type for i in issues_a] == [IssueType.DUPLICATE_TOOL_CALL]
        assert issues_b == []

        a_event_ids = {e.id for e in lens.get_events(ta.run_id)}
        b_event_ids = {e.id for e in lens.get_events(tb.run_id)}
        assert a_event_ids.isdisjoint(b_event_ids)
        assert all(i.run_id == ta.run_id for i in issues_a)
        for issue in issues_a:
            assert set(issue.related_event_ids) <= a_event_ids
    finally:
        store.close()

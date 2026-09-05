"""Integration: ``AgentLens.get_report`` over the real demo scenarios."""

import pytest

from agentlens import AgentLens
from agentlens.models import RunStatus
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import SCENARIOS, run_looping_agent, run_normal_agent


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _recount(items, key):
    counts: dict = {}
    for item in items:
        k = key(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _report_shape(report) -> dict:
    """Content that is stable across independently generated runs (no UUIDs)."""

    return {
        "status": report.run.status.value,
        "task": report.run.task,
        "event_shape": [(e.event_type.value, e.name) for e in report.events],
        "issue_types": [i.issue_type.value for i in report.issues],
        "issue_severities": [i.severity.value for i in report.issues],
        "summary": report.summary.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# Per-scenario content + summary correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_report_matches_stored_data_and_summary_is_correct(scenario_name):
    lens = AgentLens()
    run_id = SCENARIOS[scenario_name](lens)
    lens.detect(run_id)

    report = lens.get_report(run_id)

    assert report.run == lens.get_run(run_id)
    assert report.events == lens.get_events(run_id)
    assert report.issues == lens.get_issues(run_id)

    assert report.summary.total_events == len(report.events)
    assert report.summary.total_issues == len(report.issues)
    assert report.summary.events_by_type == _recount(report.events, lambda e: e.event_type)
    assert report.summary.issues_by_type == _recount(report.issues, lambda i: i.issue_type)
    assert report.summary.issues_by_severity == _recount(report.issues, lambda i: i.severity)
    assert report.run.status is RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# In-memory vs SQLite equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_in_memory_and_sqlite_reports_are_equivalent(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    mem_lens = AgentLens()
    mem_run = scenario(mem_lens)
    mem_lens.detect(mem_run)
    mem_shape = _report_shape(mem_lens.get_report(mem_run))

    store = SQLiteTraceStore(db_path)
    try:
        sql_lens = AgentLens(store=store)
        sql_run = scenario(sql_lens)
        sql_lens.detect(sql_run)
        sql_shape = _report_shape(sql_lens.get_report(sql_run))
    finally:
        store.close()

    assert sql_shape == mem_shape


# ---------------------------------------------------------------------------
# SQLite close / reopen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_report_is_identical_after_close_and_reopen(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    run_id = scenario(lens_a)
    lens_a.detect(run_id)
    before = lens_a.get_report(run_id).model_dump(mode="json")
    store_a.close()
    del lens_a, store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        lens_b = AgentLens(store=store_b)
        after = lens_b.get_report(run_id).model_dump(mode="json")
        assert after == before
        # reporting after reopen creates no issues
        issue_count = len(lens_b.get_issues(run_id))
        lens_b.get_report(run_id)
        assert len(lens_b.get_issues(run_id)) == issue_count
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_two_runs_in_one_store_produce_isolated_reports(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        loop_id = run_looping_agent(lens)
        normal_id = run_normal_agent(lens)
        lens.detect(loop_id)
        lens.detect(normal_id)

        loop_report = lens.get_report(loop_id)
        normal_report = lens.get_report(normal_id)

        assert loop_report.run.id == loop_id
        assert normal_report.run.id == normal_id
        assert all(e.run_id == loop_id for e in loop_report.events)
        assert all(e.run_id == normal_id for e in normal_report.events)
        assert all(i.run_id == loop_id for i in loop_report.issues)

        assert loop_report.summary.total_issues >= 1
        assert normal_report.summary.total_issues == 0

        loop_event_ids = {e.id for e in loop_report.events}
        normal_event_ids = {e.id for e in normal_report.events}
        assert loop_event_ids.isdisjoint(normal_event_ids)
    finally:
        store.close()


def test_separate_sqlite_files_stay_isolated_for_reports(tmp_path):
    store_a = SQLiteTraceStore(tmp_path / "a.db")
    store_b = SQLiteTraceStore(tmp_path / "b.db")
    try:
        lens_a = AgentLens(store=store_a)
        lens_b = AgentLens(store=store_b)
        a_id = run_looping_agent(lens_a)
        lens_a.detect(a_id)
        b_id = run_normal_agent(lens_b)
        lens_b.detect(b_id)

        assert lens_a.get_report(a_id).summary.total_issues >= 1
        assert lens_b.get_report(b_id).summary.total_issues == 0

        from agentlens import AgentLensError

        with pytest.raises(AgentLensError):
            lens_b.get_report(a_id)
    finally:
        store_a.close()
        store_b.close()

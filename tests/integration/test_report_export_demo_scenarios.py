"""Integration: exporting reports built from the real demo scenarios."""

import json
import re

import pytest

from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.models import AgentReport
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import SCENARIOS, run_looping_agent, run_normal_agent


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_scenario_exports_are_valid_deterministic_and_pure(scenario_name):
    lens = AgentLens()
    run_id = SCENARIOS[scenario_name](lens)
    lens.detect(run_id)
    report = lens.get_report(run_id)

    before = report.model_dump(mode="json")

    json_a = export_json(report)
    json_b = export_json(report)
    md_a = export_markdown(report)
    md_b = export_markdown(report)

    # deterministic
    assert json_a == json_b
    assert md_a == md_b

    # valid + round-trips
    parsed = json.loads(json_a)
    assert parsed == report.model_dump(mode="json")
    assert AgentReport.model_validate(parsed) == report

    # markdown has all required sections
    for heading in ["# AgentLens Report", "## Run", "## Summary", "## Events", "## Issues"]:
        assert heading in md_a

    # report unchanged, storage unchanged
    assert report.model_dump(mode="json") == before
    assert lens.get_run(run_id).model_dump(mode="json") == before["run"]
    assert lens.get_issues(run_id) == report.issues


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_exports_after_sqlite_close_and_reopen_are_identical(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    run_id = scenario(lens_a)
    lens_a.detect(run_id)
    report_before = lens_a.get_report(run_id)
    json_before = export_json(report_before)
    md_before = export_markdown(report_before)
    store_a.close()
    del lens_a, store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        lens_b = AgentLens(store=store_b)
        report_after = lens_b.get_report(run_id)
        assert export_json(report_after) == json_before
        assert export_markdown(report_after) == md_before
        # exporting created nothing
        assert len(lens_b.get_issues(run_id)) == len(report_before.issues)
    finally:
        store_b.close()


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_in_memory_and_sqlite_exports_are_equivalent(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    mem_lens = AgentLens()
    mem_report = mem_lens.get_report(_detected(mem_lens, scenario))

    store = SQLiteTraceStore(db_path)
    try:
        sql_lens = AgentLens(store=store)
        sql_report = sql_lens.get_report(_detected(sql_lens, scenario))
    finally:
        store.close()

    # Independent runs -> different UUIDs and timestamps; blind those out and the
    # remaining (scenario-deterministic) content must match exactly.
    assert _blind(export_json(mem_report)) == _blind(export_json(sql_report))
    assert _blind(export_markdown(mem_report)) == _blind(export_markdown(sql_report))


def test_undetected_run_exports_with_empty_issue_state():
    lens = AgentLens()
    run_id = run_normal_agent(lens)  # no detect
    report = lens.get_report(run_id)

    md = export_markdown(report)
    assert "_No issues detected._" in md
    parsed = json.loads(export_json(report))
    assert parsed["issues"] == []
    assert parsed["summary"]["total_issues"] == 0


def test_two_runs_export_independently(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        loop_id = run_looping_agent(lens)
        normal_id = run_normal_agent(lens)
        lens.detect(loop_id)
        lens.detect(normal_id)

        loop_md = export_markdown(lens.get_report(loop_id))
        normal_md = export_markdown(lens.get_report(normal_id))

        assert str(loop_id) in loop_md and str(normal_id) not in loop_md
        assert str(normal_id) in normal_md and str(loop_id) not in normal_md
        assert "_No issues detected._" in normal_md
        assert "_No issues detected._" not in loop_md
    finally:
        store.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _detected(lens: AgentLens, scenario):
    run_id = scenario(lens)
    lens.detect(run_id)
    return run_id


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")


def _blind(text: str) -> str:
    """Replace every UUID and ISO-8601 timestamp with a constant placeholder.

    Two independently executed runs of the same deterministic scenario differ
    only in generated UUIDs and wall-clock timestamps; everything else in the
    export is identical, so blinding those makes the exports directly comparable.
    """

    return _TS_RE.sub("<TS>", _UUID_RE.sub("<UUID>", text))

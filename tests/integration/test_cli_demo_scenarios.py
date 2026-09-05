"""Integration: the ``agentlens`` CLI over the persisted demo scenarios."""

import json
import re

import pytest

from agentlens import AgentLens
from agentlens.cli import main
from agentlens.export import export_json, export_markdown
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import SCENARIOS, run_looping_agent, run_normal_agent


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _persist_scenario(db_path, scenario):
    store = SQLiteTraceStore(db_path)
    lens = AgentLens(store=store)
    run_id = scenario(lens)
    lens.detect(run_id)
    store.close()
    return run_id


def _api_report(db_path, run_id):
    store = SQLiteTraceStore(db_path)
    try:
        return AgentLens(store=store).get_report(run_id)
    finally:
        store.close()


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_cli_over_each_demo_scenario(scenario_name, db_path, capsys):
    run_id = _persist_scenario(db_path, SCENARIOS[scenario_name])
    db = str(db_path)

    # runs: the run appears
    assert main(["runs", "--db", db]) == 0
    assert str(run_id) in capsys.readouterr().out

    # issues: count + order match persisted issues
    store = SQLiteTraceStore(db)
    persisted = AgentLens(store=store).get_issues(run_id)
    store.close()

    assert main(["issues", str(run_id), "--db", db]) == 0
    issues_out = capsys.readouterr().out
    if persisted:
        assert f"Issues: {len(persisted)}" in issues_out
        positions = [issues_out.index(str(i.id)) for i in persisted]
        assert positions == sorted(positions)
    else:
        assert issues_out == "No issues found.\n"

    # report: exactly export_markdown(get_report(...))
    expected_md = export_markdown(_api_report(db, run_id))
    assert main(["report", str(run_id), "--db", db]) == 0
    assert capsys.readouterr().out == expected_md

    # export json: parses + equals model_dump(mode="json")
    assert main(["export", str(run_id), "--format", "json", "--db", db]) == 0
    json_out = capsys.readouterr().out
    assert json.loads(json_out) == _api_report(db, run_id).model_dump(mode="json")
    assert json_out == export_json(_api_report(db, run_id))

    # export markdown: exact
    assert main(["export", str(run_id), "--format", "markdown", "--db", db]) == 0
    assert capsys.readouterr().out == expected_md


def test_cross_run_isolation(db_path, capsys):
    store = SQLiteTraceStore(db_path)
    lens = AgentLens(store=store)
    run_a = run_looping_agent(lens)
    lens.detect(run_a)
    run_b = run_normal_agent(lens)
    lens.detect(run_b)
    store.close()
    db = str(db_path)

    a_events = {str(e.id) for e in _events(db, run_a)}
    b_events = {str(e.id) for e in _events(db, run_b)}
    b_issue_ids = {str(i.id) for i in _issues(db, run_b)}

    # issues for A never mention B's ids
    main(["issues", str(run_a), "--db", db])
    out_a = capsys.readouterr().out
    assert not (b_events & _uuids_in(out_a))
    assert not (b_issue_ids & _uuids_in(out_a))
    assert str(run_b) not in out_a

    # report for A contains only A's events/issues
    main(["report", str(run_a), "--db", db])
    report_a = capsys.readouterr().out
    assert a_events & _uuids_in(report_a)
    assert not (b_events & _uuids_in(report_a))
    assert not (b_issue_ids & _uuids_in(report_a))

    # json export for A does not leak B
    main(["export", str(run_a), "--format", "json", "--db", db])
    export_a = capsys.readouterr().out
    assert not (b_events & _uuids_in(export_a))
    assert not (b_issue_ids & _uuids_in(export_a))
    assert str(run_b) not in export_a


def test_cli_after_close_and_reopen_matches_pre_close_exports(db_path, capsys):
    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    run_id = run_looping_agent(lens_a)
    lens_a.detect(run_id)
    md_before = export_markdown(lens_a.get_report(run_id))
    json_before = export_json(lens_a.get_report(run_id))
    store_a.close()
    del lens_a, store_a

    db = str(db_path)
    assert main(["export", str(run_id), "--format", "markdown", "--db", db]) == 0
    assert capsys.readouterr().out == md_before
    assert main(["export", str(run_id), "--format", "json", "--db", db]) == 0
    assert capsys.readouterr().out == json_before


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _uuids_in(text: str) -> set[str]:
    return set(_UUID_RE.findall(text))


def _events(db: str, run_id):
    store = SQLiteTraceStore(db)
    try:
        return AgentLens(store=store).get_events(run_id)
    finally:
        store.close()


def _issues(db: str, run_id):
    store = SQLiteTraceStore(db)
    try:
        return AgentLens(store=store).get_issues(run_id)
    finally:
        store.close()

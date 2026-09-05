"""Unit tests for the ``agentlens`` CLI."""

import json
from uuid import uuid4

import pytest

from agentlens import AgentLens
from agentlens.cli import build_parser, main
from agentlens.export import export_json, export_markdown
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import run_looping_agent, run_normal_agent


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


@pytest.fixture
def populated_db(db_path):
    """A SQLite DB with a detected looping run and an undetected normal run.

    Returns ``(db_path_str, loop_run_id, normal_run_id)``. The store is closed.
    """

    store = SQLiteTraceStore(db_path)
    lens = AgentLens(store=store)
    loop_id = run_looping_agent(lens)
    lens.detect(loop_id)
    normal_id = run_normal_agent(lens)  # deliberately NOT detected
    store.close()
    return str(db_path), loop_id, normal_id


def _report_via_api(db: str, run_id):
    store = SQLiteTraceStore(db)
    try:
        return AgentLens(store=store).get_report(run_id)
    finally:
        store.close()


def _snapshot(db: str):
    store = SQLiteTraceStore(db)
    try:
        lens = AgentLens(store=store)
        runs = [r.model_dump(mode="json") for r in lens.list_runs()]
        issues = {
            str(r.id): [i.model_dump(mode="json") for i in lens.get_issues(r.id)]
            for r in lens.list_runs()
        }
        events = {
            str(r.id): [e.model_dump(mode="json") for e in lens.get_events(r.id)]
            for r in lens.list_runs()
        }
        return runs, issues, events
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Parser / help
# ---------------------------------------------------------------------------


def test_root_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("runs", "issues", "report", "export"):
        assert command in out


@pytest.mark.parametrize("command", ["runs", "issues", "report", "export"])
def test_each_command_help_works(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert command in capsys.readouterr().out


def test_missing_command_errors():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_runs_requires_db():
    with pytest.raises(SystemExit) as exc:
        main(["runs"])
    assert exc.value.code != 0


def test_issues_requires_run_id_and_db():
    with pytest.raises(SystemExit):
        main(["issues", "--db", "x.db"])  # missing run_id
    with pytest.raises(SystemExit):
        main(["issues", str(uuid4())])  # missing --db


def test_export_format_is_required_and_restricted():
    with pytest.raises(SystemExit):
        main(["export", str(uuid4()), "--db", "x.db"])  # missing --format
    with pytest.raises(SystemExit):
        main(["export", str(uuid4()), "--format", "xml", "--db", "x.db"])  # bad choice


def test_build_parser_is_reusable():
    assert build_parser() is not build_parser()


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def test_runs_empty_database(db_path, capsys):
    assert main(["runs", "--db", str(db_path)]) == 0
    assert capsys.readouterr().out == "No runs found.\n"


def test_runs_lists_all_runs_in_first_save_order(populated_db, capsys):
    db, loop_id, normal_id = populated_db
    assert main(["runs", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "Runs: 2" in out
    assert out.index(str(loop_id)) < out.index(str(normal_id))  # first-save order
    assert "Task:" in out and "Status:" in out
    assert "Started At:" in out and "Finished At:" in out


# ---------------------------------------------------------------------------
# issues
# ---------------------------------------------------------------------------


def test_issues_invalid_uuid(populated_db, capsys):
    db, _, _ = populated_db
    assert main(["issues", "not-a-uuid", "--db", db]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid run id: not-a-uuid" in captured.err


def test_issues_unknown_run(populated_db, capsys):
    db, _, _ = populated_db
    missing = uuid4()
    assert main(["issues", str(missing), "--db", db]) == 1
    assert f"no run found with id {missing}" in capsys.readouterr().err


def test_issues_known_run_with_no_issues(populated_db, capsys):
    db, _, normal_id = populated_db
    assert main(["issues", str(normal_id), "--db", db]) == 0
    assert capsys.readouterr().out == "No issues found.\n"


def test_issues_output_preserves_save_order_and_deterministic_metadata(populated_db, capsys):
    db, loop_id, _ = populated_db
    store = SQLiteTraceStore(db)
    persisted = AgentLens(store=store).get_issues(loop_id)
    store.close()

    assert main(["issues", str(loop_id), "--db", db]) == 0
    out = capsys.readouterr().out
    assert f"Issues: {len(persisted)}" in out
    # each persisted issue id appears, in order
    positions = [out.index(str(i.id)) for i in persisted]
    assert positions == sorted(positions)
    # metadata rendered as sorted-key JSON
    for issue in persisted:
        rendered = json.dumps(issue.metadata, indent=2, sort_keys=True, ensure_ascii=False)
        assert rendered.splitlines()[0] in out


def _boom_detectors(*_args, **_kwargs):
    raise AssertionError("run_detectors must not be called by the CLI")


def test_issues_does_not_run_detectors(populated_db, monkeypatch, capsys):
    db, loop_id, normal_id = populated_db
    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom_detectors)
    assert main(["issues", str(normal_id), "--db", db]) == 0
    assert capsys.readouterr().out == "No issues found.\n"
    assert main(["issues", str(loop_id), "--db", db]) == 0  # a run that HAS issues
    assert "Issues:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_output_equals_export_markdown(populated_db, capsys):
    db, loop_id, _ = populated_db
    expected = export_markdown(_report_via_api(db, loop_id))
    assert main(["report", str(loop_id), "--db", db]) == 0
    assert capsys.readouterr().out == expected


def test_report_unknown_run(populated_db, capsys):
    db, _, _ = populated_db
    missing = uuid4()
    assert main(["report", str(missing), "--db", db]) == 1
    assert f"no run found with id {missing}" in capsys.readouterr().err


def test_report_invalid_uuid(populated_db, capsys):
    db, _, _ = populated_db
    assert main(["report", "nope", "--db", db]) == 1
    assert "invalid run id" in capsys.readouterr().err


def test_report_does_not_run_detectors(populated_db, monkeypatch, capsys):
    db, loop_id, _ = populated_db
    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom_detectors)
    assert main(["report", str(loop_id), "--db", db]) == 0
    assert "# AgentLens Report" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_json_is_exact_and_valid(populated_db, capsys):
    db, loop_id, _ = populated_db
    expected = export_json(_report_via_api(db, loop_id))
    assert main(["export", str(loop_id), "--format", "json", "--db", db]) == 0
    out = capsys.readouterr().out
    assert out == expected
    assert json.loads(out)  # valid JSON, no surrounding text


def test_export_markdown_is_exact(populated_db, capsys):
    db, loop_id, _ = populated_db
    expected = export_markdown(_report_via_api(db, loop_id))
    assert main(["export", str(loop_id), "--format", "markdown", "--db", db]) == 0
    out = capsys.readouterr().out
    assert out == expected
    assert not out.startswith("AgentLens CLI")


def test_export_unknown_run(populated_db, capsys):
    db, _, _ = populated_db
    missing = uuid4()
    assert main(["export", str(missing), "--format", "json", "--db", db]) == 1
    assert f"no run found with id {missing}" in capsys.readouterr().err


def test_export_does_not_run_detectors(populated_db, monkeypatch, capsys):
    db, loop_id, _ = populated_db
    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom_detectors)
    assert main(["export", str(loop_id), "--format", "json", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# read-only / determinism
# ---------------------------------------------------------------------------


def test_all_commands_are_read_only(populated_db, capsys):
    db, loop_id, normal_id = populated_db
    before = _snapshot(db)

    for argv in (
        ["runs", "--db", db],
        ["issues", str(loop_id), "--db", db],
        ["issues", str(normal_id), "--db", db],
        ["report", str(loop_id), "--db", db],
        ["export", str(loop_id), "--format", "json", "--db", db],
        ["export", str(loop_id), "--format", "markdown", "--db", db],
    ):
        main(argv)
        capsys.readouterr()

    # reopen with a fresh store and compare
    assert _snapshot(db) == before


@pytest.mark.parametrize(
    "argv_template",
    [
        ["runs"],
        ["issues", "<loop>"],
        ["report", "<loop>"],
        ["export", "<loop>", "--format", "json"],
        ["export", "<loop>", "--format", "markdown"],
    ],
)
def test_repeated_invocations_are_byte_identical(populated_db, capsys, argv_template):
    db, loop_id, _ = populated_db
    argv = [str(loop_id) if part == "<loop>" else part for part in argv_template]
    argv += ["--db", db]

    outputs = []
    for _ in range(3):
        main(argv)
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] == outputs[2]


# ---------------------------------------------------------------------------
# reopen
# ---------------------------------------------------------------------------


def test_cli_reads_persisted_data_after_store_discarded(db_path, capsys):
    store = SQLiteTraceStore(db_path)
    lens = AgentLens(store=store)
    run_id = run_looping_agent(lens)
    lens.detect(run_id)
    store.close()
    del lens, store  # nothing left alive

    assert main(["runs", "--db", str(db_path)]) == 0
    assert str(run_id) in capsys.readouterr().out
    assert main(["issues", str(run_id), "--db", str(db_path)]) == 0
    assert "Issues:" in capsys.readouterr().out
    assert main(["report", str(run_id), "--db", str(db_path)]) == 0
    assert "# AgentLens Report" in capsys.readouterr().out


def test_db_directory_that_does_not_exist_is_a_clean_error(tmp_path, capsys):
    missing = tmp_path / "nope" / "x.db"
    assert main(["runs", "--db", str(missing)]) == 1
    assert "database directory does not exist" in capsys.readouterr().err

"""Unit tests for :func:`agentlens.export.export_json`."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agentlens.export import export_json
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
        "task": "τ 任务 🚀",
        "status": RunStatus.SUCCESS,
        "started_at": _STARTED,
        "finished_at": _STARTED + timedelta(seconds=2),
        "metadata": {},
    }
    base.update(kw)
    return AgentRun(**base)


def _event(seq, **kw) -> AgentEvent:
    base = {
        "run_id": _RUN_ID,
        "sequence_number": seq,
        "event_type": EventType.DECISION,
        "name": "decide",
    }
    base.update(kw)
    return AgentEvent(**base)


def _issue(**kw) -> AgentIssue:
    base = {
        "run_id": _RUN_ID,
        "issue_type": IssueType.AGENT_LOOP,
        "severity": Severity.MEDIUM,
        "description": "loop",
    }
    base.update(kw)
    return AgentIssue(**base)


def _report(events=None, issues=None) -> AgentReport:
    return AgentReport.build(_run(), events or [], issues or [])


def test_export_json_is_valid_json():
    out = export_json(_report([_event(0)]))
    parsed = json.loads(out)
    assert isinstance(parsed, dict)


def test_parsed_output_equals_model_dump_json_mode():
    report = _report([_event(0), _event(1, event_type=EventType.RUN_COMPLETED)], [_issue()])
    parsed = json.loads(export_json(report))
    assert parsed == report.model_dump(mode="json")


def test_round_trips_through_model_validate():
    report = _report(
        [_event(0, input={"a": [1, 2, {"b": None}]}, output=True, metadata={"k": "v"})],
        [_issue(related_event_ids=[uuid4(), uuid4()])],
    )
    parsed = json.loads(export_json(report))
    assert AgentReport.model_validate(parsed) == report


def test_repeated_calls_are_byte_identical():
    report = _report([_event(0), _event(1)], [_issue(), _issue(severity=Severity.HIGH)])
    assert export_json(report) == export_json(report) == export_json(report)


def test_keys_are_sorted():
    report = _report([_event(0)])
    out = export_json(report)
    top_level_keys = list(json.loads(out).keys())
    assert top_level_keys == sorted(top_level_keys)
    # nested: run keys sorted too
    run_keys = list(json.loads(out)["run"].keys())
    assert run_keys == sorted(run_keys)


def test_unicode_is_preserved_not_escaped():
    report = _report([_event(0, name="café — 日本語", input={"emoji": "🎯"})])
    out = export_json(report)
    assert "café — 日本語" in out
    assert "🎯" in out
    assert "\\u" not in out  # ensure_ascii=False


def test_nested_values_preserved():
    payload = {"list": [1, 2.5, True, False, None, "s"], "nested": {"deep": [{"x": None}]}}
    report = _report([_event(0, input=payload, output=payload, metadata={"m": payload})])
    parsed = json.loads(export_json(report))
    assert parsed["events"][0]["input"] == payload
    assert parsed["events"][0]["metadata"]["m"] == payload


def test_uuids_and_datetimes_and_enums_serialized_as_strings():
    parsed = json.loads(export_json(_report([_event(0)], [_issue()])))
    assert isinstance(parsed["run"]["id"], str)
    assert isinstance(parsed["run"]["started_at"], str)
    assert parsed["run"]["status"] == "success"
    assert parsed["issues"][0]["issue_type"] == "agent_loop"
    assert parsed["summary"]["events_by_type"] == {"decision": 1}


def test_does_not_mutate_report():
    report = _report(
        [_event(0, input={"nested": {"a": [1]}})],
        [_issue(metadata={"detector": "x", "list": [1, 2]})],
    )
    before = report.model_dump(mode="json")
    export_json(report)
    export_json(report)
    assert report.model_dump(mode="json") == before


def test_none_values_are_handled():
    running = AgentRun(id=_RUN_ID, task="t")  # RUNNING, finished_at None
    report = AgentReport.build(
        running,
        [_event(0, input=None, output=None, duration_ms=None, status=None)],
        [],
    )
    parsed = json.loads(export_json(report))
    assert parsed["run"]["finished_at"] is None
    assert parsed["events"][0]["input"] is None
    assert parsed["events"][0]["duration_ms"] is None
    assert parsed["events"][0]["status"] is None

"""Unit tests for :meth:`agentlens.AgentLens.get_report`."""

from uuid import uuid4

import pytest

from agentlens import AgentLens, AgentLensError
from agentlens.models import AgentReport, EventType
from examples.demo_agents import run_looping_agent, run_normal_agent


def test_get_report_returns_agent_report():
    lens = AgentLens()
    run_id = run_normal_agent(lens)
    report = lens.get_report(run_id)
    assert isinstance(report, AgentReport)
    assert report.run == lens.get_run(run_id)
    assert report.events == lens.get_events(run_id)
    assert report.issues == lens.get_issues(run_id)


def test_never_detected_run_has_events_but_no_issues():
    lens = AgentLens()
    run_id = run_normal_agent(lens)  # no detect() call
    report = lens.get_report(run_id)

    assert report.events  # normal scenario records events
    assert report.issues == []
    assert report.summary.total_issues == 0
    assert report.summary.issues_by_type == {}
    assert report.summary.issues_by_severity == {}
    assert report.summary.total_events == len(report.events)


def test_summary_totals_match_collections():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    lens.detect(run_id)
    report = lens.get_report(run_id)

    assert report.summary.total_events == len(report.events)
    assert report.summary.total_issues == len(report.issues)
    # independently recomputed groupings
    events_by_type: dict = {}
    for e in report.events:
        events_by_type[e.event_type] = events_by_type.get(e.event_type, 0) + 1
    assert report.summary.events_by_type == events_by_type


def test_repeated_detection_is_not_deduplicated_in_the_report():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    first = lens.detect(run_id)
    second = lens.detect(run_id)

    report = lens.get_report(run_id)
    assert report.issues == lens.get_issues(run_id)
    assert len(report.issues) == len(first) + len(second)
    assert report.summary.total_issues == len(first) + len(second)
    # both batches present, in save order
    assert report.issues[: len(first)] == first
    assert report.issues[len(first) :] == second


def test_get_report_is_read_only():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    lens.detect(run_id)

    run_before = lens.get_run(run_id).model_dump(mode="json")
    events_before = [e.model_dump(mode="json") for e in lens.get_events(run_id)]
    issues_before = [i.model_dump(mode="json") for i in lens.get_issues(run_id)]

    lens.get_report(run_id)
    lens.get_report(run_id)
    lens.get_report(run_id)

    assert lens.get_run(run_id).model_dump(mode="json") == run_before
    assert [e.model_dump(mode="json") for e in lens.get_events(run_id)] == events_before
    assert [i.model_dump(mode="json") for i in lens.get_issues(run_id)] == issues_before


def test_repeated_get_report_is_deterministic():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    lens.detect(run_id)

    a = lens.get_report(run_id).model_dump(mode="json")
    b = lens.get_report(run_id).model_dump(mode="json")
    c = lens.get_report(run_id).model_dump(mode="json")
    assert a == b == c


def test_get_report_does_not_run_detectors(monkeypatch):
    lens = AgentLens()
    run_id = run_looping_agent(lens)

    def _boom(*_args, **_kwargs):
        raise AssertionError("get_report must not call run_detectors")

    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom)

    report = lens.get_report(run_id)  # must not raise
    assert report.issues == []  # nothing detected, nothing persisted
    assert lens.get_issues(run_id) == []


def test_unknown_run_raises_agentlens_error():
    lens = AgentLens()
    unknown = uuid4()
    with pytest.raises(AgentLensError, match="no run with id"):
        lens.get_report(unknown)
    # storage untouched
    assert lens.list_runs() == []


def test_lifecycle_events_are_included_in_the_count():
    lens = AgentLens()
    with lens.trace("t") as trace:
        trace.record_event(event_type=EventType.DECISION, name="d")
    report = lens.get_report(trace.run_id)
    # RUN_STARTED + DECISION + RUN_COMPLETED
    assert report.summary.total_events == 3
    assert report.summary.events_by_type[EventType.RUN_STARTED] == 1
    assert report.summary.events_by_type[EventType.RUN_COMPLETED] == 1

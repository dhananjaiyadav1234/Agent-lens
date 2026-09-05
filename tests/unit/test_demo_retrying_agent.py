"""Trace-level tests for the excessive-retry demo scenario."""

import json

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import run_retrying_agent


def test_run_completes_successfully_without_context_failure():
    lens = AgentLens()
    run = lens.get_run(run_retrying_agent(lens))
    # the with-block never raised: the run is SUCCESS, not FAILED
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None


def test_multiple_attempts_with_at_least_three_failures_then_success():
    lens = AgentLens()
    events = lens.get_events(run_retrying_agent(lens))

    starts = [e for e in events if e.event_type is EventType.TOOL_CALL_STARTED]
    errors = [e for e in events if e.event_type is EventType.ERROR]
    completed = [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]

    assert len(starts) >= 4
    assert len(errors) >= 3
    assert len(completed) == 1
    # the successful completion is the last tool event
    tool_events = [e for e in events if e.name in {"fetch_customer"}]
    assert tool_events[-1].event_type is EventType.TOOL_CALL_COMPLETED


def test_all_attempts_are_the_same_logical_operation():
    lens = AgentLens()
    events = lens.get_events(run_retrying_agent(lens))
    attempt_events = [
        e
        for e in events
        if e.event_type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
    ]
    assert {e.name for e in attempt_events} == {"fetch_customer"}
    assert all(e.input == {"customer_id": "123"} for e in attempt_events)


def test_failures_are_structured_json_error_events():
    lens = AgentLens()
    events = lens.get_events(run_retrying_agent(lens))
    errors = [e for e in events if e.event_type is EventType.ERROR]

    assert len(errors) == 3
    for e in errors:
        assert e.status == "error"
        assert e.name == "temporary_tool_failure"
        assert e.output["operation"] == "fetch_customer"
        assert e.output["error_type"] == "TemporaryError"
        assert isinstance(e.output["message"], str)
        # payload is plain JSON
        assert json.loads(json.dumps(e.output)) == e.output


def test_deterministic_event_pattern_across_repeat_runs():
    lens = AgentLens()
    first = [(e.event_type, e.name) for e in lens.get_events(run_retrying_agent(lens))]
    second = [(e.event_type, e.name) for e in lens.get_events(run_retrying_agent(lens))]
    assert first == second

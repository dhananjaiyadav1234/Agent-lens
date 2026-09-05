"""Trace-level tests for the duplicate-tool-call demo scenario."""

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import run_duplicate_tool_agent


def _tool_starts(events):
    return [e for e in events if e.event_type is EventType.TOOL_CALL_STARTED]


def test_run_completes_successfully():
    lens = AgentLens()
    run = lens.get_run(run_duplicate_tool_agent(lens))
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None


def test_duplicate_tool_call_exists_with_same_name_and_input():
    lens = AgentLens()
    events = lens.get_events(run_duplicate_tool_agent(lens))
    starts = _tool_starts(events)

    assert len(starts) == 2
    assert starts[0].name == starts[1].name == "lookup_customer"
    assert starts[0].input == starts[1].input == {"customer_id": "123"}


def test_completed_calls_also_match():
    lens = AgentLens()
    events = lens.get_events(run_duplicate_tool_agent(lens))
    completed = [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 2
    assert completed[0].name == completed[1].name == "lookup_customer"
    assert completed[0].input == completed[1].input
    assert completed[0].output == completed[1].output


def test_second_call_is_marked_as_not_depending_on_new_information():
    lens = AgentLens()
    events = lens.get_events(run_duplicate_tool_agent(lens))
    recheck = next(e for e in events if e.name == "double_check_customer_details")
    assert recheck.output["uses_new_information"] is False
    second_start = _tool_starts(events)[1]
    assert second_start.metadata.get("depends_on_new_information") is False


def test_scenario_is_deterministic():
    lens = AgentLens()
    first = [
        (e.event_type, e.name, e.input) for e in lens.get_events(run_duplicate_tool_agent(lens))
    ]
    second = [
        (e.event_type, e.name, e.input) for e in lens.get_events(run_duplicate_tool_agent(lens))
    ]
    assert first == second

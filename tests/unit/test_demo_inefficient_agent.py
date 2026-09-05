"""Trace-level tests for the inefficient (over-retrieval) demo scenario."""

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import run_inefficient_agent


def test_run_completes_successfully():
    lens = AgentLens()
    run = lens.get_run(run_inefficient_agent(lens))
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None


def test_unnecessary_broad_retrieval_is_present():
    lens = AgentLens()
    events = lens.get_events(run_inefficient_agent(lens))
    tool_names = [e.name for e in events if e.event_type is EventType.TOOL_CALL_STARTED]
    # a targeted lookup was enough, but the agent also fetched everything
    assert "lookup_customer" in tool_names
    assert "fetch_all_customers" in tool_names


def test_metadata_describes_purpose_and_scope_gap():
    lens = AgentLens()
    events = lens.get_events(run_inefficient_agent(lens))
    broad = next(
        e
        for e in events
        if e.name == "fetch_all_customers" and e.event_type is EventType.TOOL_CALL_STARTED
    )
    assert broad.metadata["purpose"] == "retrieve all customers"
    assert broad.metadata["required_scope"] == "single_customer"
    assert broad.metadata["actual_scope"] == "all_customers"

    realization = next(e for e in events if e.name == "recognize_over_retrieval")
    assert realization.metadata["wasted_operation"] == "fetch_all_customers"


def test_not_a_loop_or_retry_or_duplicate_tool_scenario():
    lens = AgentLens()
    events = lens.get_events(run_inefficient_agent(lens))

    # no ERROR events -> not the retry scenario
    assert not [e for e in events if e.event_type is EventType.ERROR]

    # every tool-call-started has a distinct (name, input) -> not a duplicate-tool scenario
    starts = [
        (e.name, tuple(sorted((e.input or {}).items())))
        for e in events
        if e.event_type is EventType.TOOL_CALL_STARTED
    ]
    assert len(starts) == len(set(starts))

    # no decision name repeats -> not the looping scenario
    decisions = [e.name for e in events if e.event_type is EventType.DECISION]
    assert len(decisions) == len(set(decisions))


def test_event_sequence_is_deterministic():
    lens = AgentLens()
    first = [(e.event_type, e.name) for e in lens.get_events(run_inefficient_agent(lens))]
    second = [(e.event_type, e.name) for e in lens.get_events(run_inefficient_agent(lens))]
    assert first == second
    assert first[0][0] is EventType.RUN_STARTED
    assert first[-1][0] is EventType.RUN_COMPLETED

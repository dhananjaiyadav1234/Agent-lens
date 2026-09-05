"""Trace-level tests for the healthy (normal) demo scenario."""

from collections import Counter

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import run_normal_agent


def _events(lens, run_id):
    return lens.get_events(run_id)


def test_run_completes_successfully():
    lens = AgentLens()
    run_id = run_normal_agent(lens)
    run = lens.get_run(run_id)
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None


def test_expected_event_pattern():
    lens = AgentLens()
    run_id = run_normal_agent(lens)
    pattern = [(e.event_type, e.name) for e in _events(lens, run_id)]
    assert pattern == [
        (EventType.RUN_STARTED, "run_started"),
        (EventType.DECISION, "choose_lookup_strategy"),
        (EventType.TOOL_CALL_STARTED, "lookup_customer"),
        (EventType.TOOL_CALL_COMPLETED, "lookup_customer"),
        (EventType.DECISION, "prepare_response"),
        (EventType.RUN_COMPLETED, "run_completed"),
    ]


def test_no_repeated_decision_cycle():
    lens = AgentLens()
    run_id = run_normal_agent(lens)
    decision_names = [e.name for e in _events(lens, run_id) if e.event_type is EventType.DECISION]
    assert len(decision_names) == len(set(decision_names))


def test_no_duplicate_tool_call():
    lens = AgentLens()
    run_id = run_normal_agent(lens)
    starts = [
        (e.name, e.input)
        for e in _events(lens, run_id)
        if e.event_type is EventType.TOOL_CALL_STARTED
    ]
    assert len(starts) == 1
    assert Counter(name for name, _ in starts)["lookup_customer"] == 1


def test_events_are_ordered_and_contiguous():
    lens = AgentLens()
    run_id = run_normal_agent(lens)
    seqs = [e.sequence_number for e in _events(lens, run_id)]
    assert seqs == list(range(len(seqs)))


def test_deterministic_event_names_across_repeat_runs():
    lens = AgentLens()
    first = [e.name for e in _events(lens, run_normal_agent(lens))]
    second = [e.name for e in _events(lens, run_normal_agent(lens))]
    assert first == second

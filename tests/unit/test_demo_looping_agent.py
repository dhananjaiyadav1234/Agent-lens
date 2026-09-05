"""Trace-level tests for the looping demo scenario."""

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import run_looping_agent

_CYCLE = ["analyze_request", "search_for_information"]


def test_run_completes_successfully():
    lens = AgentLens()
    run = lens.get_run(run_looping_agent(lens))
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None


def test_repeated_cycle_occurs_at_least_three_times():
    lens = AgentLens()
    events = lens.get_events(run_looping_agent(lens))
    names = [e.name for e in events if e.event_type is EventType.DECISION]

    cycles = 0
    i = 0
    while i + len(_CYCLE) <= len(names):
        if names[i : i + len(_CYCLE)] == _CYCLE:
            cycles += 1
            i += len(_CYCLE)
        else:
            i += 1
    assert cycles >= 3


def test_repeated_events_are_deterministic_and_use_identical_input():
    lens = AgentLens()
    events = lens.get_events(run_looping_agent(lens))

    analyze_inputs = [e.input for e in events if e.name == "analyze_request"]
    search_inputs = [e.input for e in events if e.name == "search_for_information"]

    assert len(analyze_inputs) >= 3
    assert len(search_inputs) >= 3
    assert all(payload == analyze_inputs[0] for payload in analyze_inputs)
    assert all(payload == search_inputs[0] for payload in search_inputs)


def test_sequence_numbers_are_contiguous_and_unique():
    lens = AgentLens()
    seqs = [e.sequence_number for e in lens.get_events(run_looping_agent(lens))]
    assert seqs == list(range(len(seqs)))
    assert len(seqs) == len(set(seqs))


def test_scenario_terminates_with_an_exit_decision():
    lens = AgentLens()
    events = lens.get_events(run_looping_agent(lens))
    names = [e.name for e in events]
    assert "abandon_search_and_respond" in names
    # the exit decision comes after the loop and before RUN_COMPLETED
    assert names.index("abandon_search_and_respond") > names.index("search_for_information")
    assert names[-1] == "run_completed"


def test_iteration_metadata_counts_up():
    lens = AgentLens()
    events = lens.get_events(run_looping_agent(lens))
    iters = [e.metadata["iteration"] for e in events if e.name == "analyze_request"]
    assert iters == sorted(iters)
    assert iters[0] == 0

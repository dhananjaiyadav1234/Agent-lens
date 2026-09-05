"""Integration: LoopDetector over the real demo scenarios.

Full path -- demo agent -> AgentLens public API -> trace context -> event
recording -> in-memory store -> LoopDetector -> AgentIssue. No models are
constructed by hand.
"""

import json

import pytest

from agentlens import AgentLens
from agentlens.detectors import LoopDetector
from agentlens.models import EventType, IssueType, Severity
from examples.demo_agents import (
    run_duplicate_tool_agent,
    run_inefficient_agent,
    run_looping_agent,
    run_normal_agent,
    run_retrying_agent,
)


@pytest.mark.parametrize(
    "scenario",
    [run_normal_agent, run_retrying_agent, run_duplicate_tool_agent, run_inefficient_agent],
    ids=["normal", "retrying", "duplicate", "inefficient"],
)
def test_non_looping_scenarios_produce_no_loop_issues(scenario):
    lens = AgentLens()
    run_id = scenario(lens)
    issues = LoopDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
    assert issues == []


def test_looping_demo_produces_exactly_one_medium_loop_issue():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    run = lens.get_run(run_id)
    events = lens.get_events(run_id)

    issues = LoopDetector().detect(run, events)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type is IssueType.LOOP
    assert issue.severity is Severity.MEDIUM
    assert issue.metadata["cycle_length"] == 2
    assert issue.metadata["repetitions"] == 3
    assert issue.metadata["pattern"] == [
        {"event_type": "decision", "name": "analyze_request"},
        {"event_type": "decision", "name": "search_for_information"},
    ]


def test_loop_issue_references_exactly_the_six_cycle_events():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    events = lens.get_events(run_id)
    issue = LoopDetector().detect(lens.get_run(run_id), events)[0]

    loop_ids = issue.related_event_ids
    assert len(loop_ids) == 6

    by_id = {e.id: e for e in events}
    loop_events = [by_id[eid] for eid in loop_ids]

    # order preserved, only analyze/search decisions, no lifecycle or exit event
    assert [e.name for e in loop_events] == [
        "analyze_request",
        "search_for_information",
    ] * 3
    assert all(e.event_type is EventType.DECISION for e in loop_events)

    excluded_names = {"run_started", "abandon_search_and_respond", "run_completed"}
    assert excluded_names.isdisjoint({e.name for e in loop_events})


def test_issue_run_id_matches_and_round_trips():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    run = lens.get_run(run_id)
    issue = LoopDetector().detect(run, lens.get_events(run_id))[0]

    assert issue.run_id == run.id
    json.dumps(issue.metadata)  # metadata is JSON-compatible
    from agentlens.models import AgentIssue

    assert AgentIssue.model_validate_json(issue.model_dump_json()) == issue


def test_no_event_leakage_between_runs_on_one_lens():
    lens = AgentLens()
    loop_id = run_looping_agent(lens)
    normal_id = run_normal_agent(lens)

    loop_issue = LoopDetector().detect(lens.get_run(loop_id), lens.get_events(loop_id))[0]

    loop_run_event_ids = {e.id for e in lens.get_events(loop_id)}
    normal_run_event_ids = {e.id for e in lens.get_events(normal_id)}

    assert set(loop_issue.related_event_ids) <= loop_run_event_ids
    assert set(loop_issue.related_event_ids).isdisjoint(normal_run_event_ids)
    assert LoopDetector().detect(lens.get_run(normal_id), lens.get_events(normal_id)) == []


def test_detector_does_not_disturb_stored_events():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    events = lens.get_events(run_id)

    before = [e.model_dump() for e in events]
    order_before = [e.sequence_number for e in events]

    LoopDetector().detect(lens.get_run(run_id), events)

    assert [e.model_dump() for e in events] == before
    assert [e.sequence_number for e in events] == order_before
    assert [e.sequence_number for e in lens.get_events(run_id)] == order_before

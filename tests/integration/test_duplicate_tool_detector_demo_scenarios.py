"""Integration: DuplicateToolDetector over the real demo scenarios."""

import json

import pytest

from agentlens import AgentLens
from agentlens.detectors import DuplicateToolDetector
from agentlens.models import AgentIssue, EventType, IssueType, RunStatus, Severity
from examples.demo_agents import (
    run_duplicate_tool_agent,
    run_inefficient_agent,
    run_looping_agent,
    run_normal_agent,
    run_retrying_agent,
)


@pytest.mark.parametrize(
    "scenario",
    [run_normal_agent, run_looping_agent, run_retrying_agent, run_inefficient_agent],
    ids=["normal", "looping", "retrying", "inefficient"],
)
def test_non_duplicate_scenarios_produce_no_issues(scenario):
    lens = AgentLens()
    run_id = scenario(lens)
    issues = DuplicateToolDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
    assert issues == []


def test_duplicate_demo_produces_exactly_one_medium_issue():
    lens = AgentLens()
    run_id = run_duplicate_tool_agent(lens)
    run = lens.get_run(run_id)
    events = lens.get_events(run_id)

    issues = DuplicateToolDetector().detect(run, events)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type is IssueType.DUPLICATE
    assert issue.severity is Severity.MEDIUM
    assert issue.metadata["operation"] == "lookup_customer"
    assert issue.metadata["input"] == {"customer_id": "123"}
    assert issue.metadata["duplicate_count"] == 1
    assert run.status is RunStatus.SUCCESS


def test_duplicate_issue_references_both_start_completed_pairs_only():
    lens = AgentLens()
    run_id = run_duplicate_tool_agent(lens)
    events = lens.get_events(run_id)
    issue = DuplicateToolDetector().detect(lens.get_run(run_id), events)[0]

    related = issue.related_event_ids
    assert len(related) == 4
    assert len(set(related)) == 4

    by_id = {e.id: e for e in events}
    related_events = [by_id[eid] for eid in related]

    assert [e.event_type for e in related_events] == [
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
    ]
    assert all(e.name == "lookup_customer" for e in related_events)

    # trace order
    seqs = [e.sequence_number for e in related_events]
    assert seqs == sorted(seqs)

    # no lifecycle, no decisions
    lifecycle_ids = {
        e.id for e in events if e.event_type in {EventType.RUN_STARTED, EventType.RUN_COMPLETED}
    }
    decision_ids = {e.id for e in events if e.event_type is EventType.DECISION}
    assert lifecycle_ids.isdisjoint(related)
    assert decision_ids.isdisjoint(related)


def test_first_lookup_is_not_flagged():
    lens = AgentLens()
    run_id = run_duplicate_tool_agent(lens)
    events = lens.get_events(run_id)
    issue = DuplicateToolDetector().detect(lens.get_run(run_id), events)[0]

    starts = [e for e in events if e.event_type is EventType.TOOL_CALL_STARTED]
    assert len(starts) == 2
    # the flagged duplicate start is the second one
    assert issue.metadata["duplicate_sequence_number"] == starts[1].sequence_number
    assert issue.metadata["original_sequence_number"] == starts[0].sequence_number


def test_issue_json_round_trips_and_run_id_matches():
    lens = AgentLens()
    run_id = run_duplicate_tool_agent(lens)
    run = lens.get_run(run_id)
    issue = DuplicateToolDetector().detect(run, lens.get_events(run_id))[0]

    assert issue.run_id == run.id
    json.dumps(issue.metadata)
    assert AgentIssue.model_validate_json(issue.model_dump_json()) == issue


def test_detector_does_not_mutate_stored_events():
    lens = AgentLens()
    run_id = run_duplicate_tool_agent(lens)
    events = lens.get_events(run_id)

    before = [e.model_dump() for e in events]
    DuplicateToolDetector().detect(lens.get_run(run_id), events)
    assert [e.model_dump() for e in events] == before
    assert [e.sequence_number for e in lens.get_events(run_id)] == [
        e.sequence_number for e in events
    ]


def test_no_state_leakage_between_runs_on_one_lens():
    lens = AgentLens()
    dup_id = run_duplicate_tool_agent(lens)
    normal_id = run_normal_agent(lens)

    dup_issue = DuplicateToolDetector().detect(lens.get_run(dup_id), lens.get_events(dup_id))[0]

    dup_event_ids = {e.id for e in lens.get_events(dup_id)}
    normal_event_ids = {e.id for e in lens.get_events(normal_id)}
    assert set(dup_issue.related_event_ids) <= dup_event_ids
    assert set(dup_issue.related_event_ids).isdisjoint(normal_event_ids)

    assert DuplicateToolDetector().detect(lens.get_run(normal_id), lens.get_events(normal_id)) == []


def test_isolation_across_two_lens_instances():
    lens_a = AgentLens()
    lens_b = AgentLens()

    a_id = run_duplicate_tool_agent(lens_a)
    b_id = run_duplicate_tool_agent(lens_b)

    a_issue = DuplicateToolDetector().detect(lens_a.get_run(a_id), lens_a.get_events(a_id))[0]
    b_issue = DuplicateToolDetector().detect(lens_b.get_run(b_id), lens_b.get_events(b_id))[0]

    assert a_issue.run_id != b_issue.run_id
    assert set(a_issue.related_event_ids).isdisjoint(b_issue.related_event_ids)
    assert a_issue.metadata["operation"] == b_issue.metadata["operation"] == "lookup_customer"

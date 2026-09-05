"""Integration: InefficiencyDetector over the real demo scenarios."""

import json

import pytest

from agentlens import AgentLens
from agentlens.detectors import InefficiencyDetector
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
    [run_normal_agent, run_looping_agent, run_retrying_agent, run_duplicate_tool_agent],
    ids=["normal", "looping", "retrying", "duplicate"],
)
def test_non_inefficient_scenarios_produce_no_issues(scenario):
    lens = AgentLens()
    run_id = scenario(lens)
    issues = InefficiencyDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
    assert issues == []


def test_inefficient_demo_produces_exactly_one_issue():
    lens = AgentLens()
    run_id = run_inefficient_agent(lens)
    run = lens.get_run(run_id)
    events = lens.get_events(run_id)

    issues = InefficiencyDetector().detect(run, events)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.run_id == run.id
    assert issue.issue_type is IssueType.INEFFICIENCY
    assert issue.metadata["operation"] == "fetch_all_customers"
    assert issue.metadata["input"] == {}
    assert issue.metadata["evidence"] == [
        "scope_mismatch",
        "explicit_wasted_operation",
        "sufficient_information_already_available",
    ]
    assert issue.severity is Severity.CRITICAL  # 3 independent explicit signals
    assert run.status is RunStatus.SUCCESS


def test_inefficient_demo_related_events():
    lens = AgentLens()
    run_id = run_inefficient_agent(lens)
    events = lens.get_events(run_id)
    issue = InefficiencyDetector().detect(lens.get_run(run_id), events)[0]

    by_id = {e.id: e for e in events}
    related = [by_id[i] for i in issue.related_event_ids]

    # the broad successful tool call is referenced
    kinds = [(e.event_type, e.name) for e in related]
    assert (EventType.TOOL_CALL_STARTED, "fetch_all_customers") in kinds
    assert (EventType.TOOL_CALL_COMPLETED, "fetch_all_customers") in kinds
    # the explicit recognition and the sufficiency event
    assert any("wasted_operation" in (e.metadata or {}) for e in related)
    assert any((e.metadata or {}).get("sufficient_to_answer") is True for e in related)

    # unique, trace-ordered, no lifecycle
    assert len(issue.related_event_ids) == len(set(issue.related_event_ids)) == 4
    seqs = [e.sequence_number for e in related]
    assert seqs == sorted(seqs)
    lifecycle_ids = {
        e.id for e in events if e.event_type in {EventType.RUN_STARTED, EventType.RUN_COMPLETED}
    }
    assert lifecycle_ids.isdisjoint(issue.related_event_ids)

    # the initial targeted lookup is NOT referenced as inefficient work
    assert (EventType.TOOL_CALL_STARTED, "lookup_customer") not in kinds


def test_initial_targeted_lookup_is_not_flagged():
    lens = AgentLens()
    run_id = run_inefficient_agent(lens)
    issues = InefficiencyDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
    assert [i.metadata["operation"] for i in issues] == ["fetch_all_customers"]


def test_issue_json_round_trips():
    lens = AgentLens()
    run_id = run_inefficient_agent(lens)
    run = lens.get_run(run_id)
    issue = InefficiencyDetector().detect(run, lens.get_events(run_id))[0]
    json.dumps(issue.metadata)
    assert AgentIssue.model_validate_json(issue.model_dump_json()) == issue


def test_detector_does_not_mutate_stored_events():
    lens = AgentLens()
    run_id = run_inefficient_agent(lens)
    events = lens.get_events(run_id)
    before = [e.model_dump() for e in events]
    InefficiencyDetector().detect(lens.get_run(run_id), events)
    assert [e.model_dump() for e in events] == before
    assert [e.sequence_number for e in lens.get_events(run_id)] == [
        e.sequence_number for e in events
    ]


def test_no_state_leakage_between_runs_on_one_lens():
    lens = AgentLens()
    ineff_id = run_inefficient_agent(lens)
    normal_id = run_normal_agent(lens)

    issue = InefficiencyDetector().detect(lens.get_run(ineff_id), lens.get_events(ineff_id))[0]
    ineff_ids = {e.id for e in lens.get_events(ineff_id)}
    normal_ids = {e.id for e in lens.get_events(normal_id)}
    assert set(issue.related_event_ids) <= ineff_ids
    assert set(issue.related_event_ids).isdisjoint(normal_ids)
    assert InefficiencyDetector().detect(lens.get_run(normal_id), lens.get_events(normal_id)) == []


def test_isolation_across_two_lens_instances():
    lens_a = AgentLens()
    lens_b = AgentLens()
    a_id = run_inefficient_agent(lens_a)
    b_id = run_inefficient_agent(lens_b)

    a_issue = InefficiencyDetector().detect(lens_a.get_run(a_id), lens_a.get_events(a_id))[0]
    b_issue = InefficiencyDetector().detect(lens_b.get_run(b_id), lens_b.get_events(b_id))[0]

    assert a_issue.run_id != b_issue.run_id
    assert set(a_issue.related_event_ids).isdisjoint(b_issue.related_event_ids)
    assert a_issue.metadata == {**b_issue.metadata}  # equivalent content

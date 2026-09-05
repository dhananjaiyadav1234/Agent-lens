"""Integration: RetryDetector over the real demo scenarios.

Full path -- demo agent -> AgentLens public API -> trace context -> event
recording -> in-memory store -> RetryDetector -> AgentIssue. No models are
constructed by hand.
"""

import json

import pytest

from agentlens import AgentLens
from agentlens.detectors import RetryDetector
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
    [run_normal_agent, run_looping_agent, run_duplicate_tool_agent, run_inefficient_agent],
    ids=["normal", "looping", "duplicate", "inefficient"],
)
def test_non_retry_scenarios_produce_no_retry_issues(scenario):
    lens = AgentLens()
    run_id = scenario(lens)
    issues = RetryDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
    assert issues == []


def test_retrying_demo_produces_exactly_one_medium_retry_issue():
    lens = AgentLens()
    run_id = run_retrying_agent(lens)
    run = lens.get_run(run_id)
    events = lens.get_events(run_id)

    issues = RetryDetector().detect(run, events)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type is IssueType.RETRY
    assert issue.severity is Severity.MEDIUM
    assert issue.metadata["operation"] == "fetch_customer"
    assert issue.metadata["failed_attempts"] == 3
    assert issue.metadata["input"] == {"customer_id": "123"}
    assert run.status is RunStatus.SUCCESS


def test_retry_issue_references_the_three_failed_attempts_only():
    lens = AgentLens()
    run_id = run_retrying_agent(lens)
    events = lens.get_events(run_id)
    issue = RetryDetector().detect(lens.get_run(run_id), events)[0]

    related = issue.related_event_ids
    assert len(related) == 6

    by_id = {e.id: e for e in events}
    related_events = [by_id[eid] for eid in related]

    assert [e.event_type for e in related_events] == [
        EventType.TOOL_CALL_STARTED,
        EventType.ERROR,
        EventType.TOOL_CALL_STARTED,
        EventType.ERROR,
        EventType.TOOL_CALL_STARTED,
        EventType.ERROR,
    ]
    assert all(
        e.name == "fetch_customer"
        for e in related_events
        if e.event_type is EventType.TOOL_CALL_STARTED
    )

    # the final, successful attempt is excluded
    completed = [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 1
    assert completed[0].id not in related

    # lifecycle events excluded
    lifecycle_ids = {
        e.id for e in events if e.event_type in {EventType.RUN_STARTED, EventType.RUN_COMPLETED}
    }
    assert lifecycle_ids.isdisjoint(related)


def test_issue_run_id_matches_and_json_round_trips():
    lens = AgentLens()
    run_id = run_retrying_agent(lens)
    run = lens.get_run(run_id)
    issue = RetryDetector().detect(run, lens.get_events(run_id))[0]

    assert issue.run_id == run.id
    json.dumps(issue.metadata)
    assert AgentIssue.model_validate_json(issue.model_dump_json()) == issue


def test_no_event_leakage_between_two_runs_on_one_lens():
    lens = AgentLens()
    retry_id = run_retrying_agent(lens)
    normal_id = run_normal_agent(lens)

    issue = RetryDetector().detect(lens.get_run(retry_id), lens.get_events(retry_id))[0]

    retry_event_ids = {e.id for e in lens.get_events(retry_id)}
    normal_event_ids = {e.id for e in lens.get_events(normal_id)}

    assert set(issue.related_event_ids) <= retry_event_ids
    assert set(issue.related_event_ids).isdisjoint(normal_event_ids)
    assert RetryDetector().detect(lens.get_run(normal_id), lens.get_events(normal_id)) == []


def test_detector_does_not_disturb_stored_events():
    lens = AgentLens()
    run_id = run_retrying_agent(lens)
    events = lens.get_events(run_id)

    before = [e.model_dump() for e in events]
    RetryDetector().detect(lens.get_run(run_id), events)

    assert [e.model_dump() for e in events] == before
    assert [e.sequence_number for e in lens.get_events(run_id)] == [
        e.sequence_number for e in events
    ]

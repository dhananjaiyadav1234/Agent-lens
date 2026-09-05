"""Integration: ``AgentLens.detect`` over the real demo scenarios."""

import pytest

from agentlens import AgentLens, AgentLensError
from agentlens.detectors import (
    DuplicateToolDetector,
    InefficiencyDetector,
    LoopDetector,
    RetryDetector,
)
from agentlens.models import AgentIssue, IssueType, RunStatus
from examples.demo_agents import (
    SCENARIOS,
    run_looping_agent,
    run_normal_agent,
)


def _content(issue: AgentIssue) -> dict:
    return issue.model_dump(mode="json", exclude={"id"})


def _manual(run, events) -> list[AgentIssue]:
    return [
        *LoopDetector().detect(run, events),
        *RetryDetector().detect(run, events),
        *DuplicateToolDetector().detect(run, events),
        *InefficiencyDetector().detect(run, events),
    ]


@pytest.mark.parametrize(
    ("scenario_name", "expected_types"),
    [
        ("normal", []),
        ("looping", [IssueType.AGENT_LOOP]),
        ("retrying", [IssueType.EXCESSIVE_RETRY]),
        ("duplicate", [IssueType.DUPLICATE_TOOL_CALL]),
        ("inefficient", [IssueType.INEFFICIENCY]),
    ],
)
def test_each_demo_scenario_yields_only_its_expected_issue_types(scenario_name, expected_types):
    lens = AgentLens()
    run_id = SCENARIOS[scenario_name](lens)
    issues = lens.detect(run_id)
    assert [i.issue_type for i in issues] == expected_types
    assert all(i.run_id == run_id for i in issues)


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_detect_matches_manual_concatenation_for_every_scenario(scenario_name):
    lens = AgentLens()
    run_id = SCENARIOS[scenario_name](lens)
    run = lens.get_run(run_id)
    events = lens.get_events(run_id)

    orchestrated = lens.detect(run_id)
    manual = _manual(run, events)

    assert [_content(i) for i in orchestrated] == [_content(i) for i in manual]


def test_runs_stay_success_and_events_untouched():
    lens = AgentLens()
    for scenario in SCENARIOS.values():
        run_id = scenario(lens)
        events_before = [e.model_dump() for e in lens.get_events(run_id)]

        lens.detect(run_id)

        assert lens.get_run(run_id).status is RunStatus.SUCCESS
        assert [e.model_dump() for e in lens.get_events(run_id)] == events_before


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_two_runs_on_one_lens_do_not_leak_events():
    lens = AgentLens()
    loop_id = run_looping_agent(lens)
    normal_id = run_normal_agent(lens)

    loop_issues = lens.detect(loop_id)
    normal_issues = lens.detect(normal_id)

    loop_event_ids = {e.id for e in lens.get_events(loop_id)}
    normal_event_ids = {e.id for e in lens.get_events(normal_id)}

    assert normal_issues == []
    assert loop_issues and all(set(i.related_event_ids) <= loop_event_ids for i in loop_issues)
    assert all(set(i.related_event_ids).isdisjoint(normal_event_ids) for i in loop_issues)


def test_detecting_one_run_does_not_change_another():
    lens = AgentLens()
    a_id = run_looping_agent(lens)
    b_id = run_looping_agent(lens)

    b_before = [e.model_dump() for e in lens.get_events(b_id)]
    lens.detect(a_id)
    lens.detect(a_id)
    assert [e.model_dump() for e in lens.get_events(b_id)] == b_before

    # b still detects the same thing
    assert [_content(i) for i in lens.detect(b_id)] == [
        _content(i) for i in _manual(lens.get_run(b_id), lens.get_events(b_id))
    ]


def test_separate_lens_instances_are_isolated():
    lens_a = AgentLens()
    lens_b = AgentLens()
    a_id = run_looping_agent(lens_a)
    b_id = run_looping_agent(lens_b)

    assert lens_b.detect(b_id)  # b works on its own instance
    with pytest.raises(AgentLensError):
        lens_a.detect(b_id)  # b's run is unknown to lens_a

    a_issue = lens_a.detect(a_id)[0]
    b_issue = lens_b.detect(b_id)[0]
    assert a_issue.run_id != b_issue.run_id
    assert set(a_issue.related_event_ids).isdisjoint(b_issue.related_event_ids)


def test_repeated_detect_calls_return_equivalent_content():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    first = [_content(i) for i in lens.detect(run_id)]
    second = [_content(i) for i in lens.detect(run_id)]
    third = [_content(i) for i in lens.detect(run_id)]
    assert first == second == third

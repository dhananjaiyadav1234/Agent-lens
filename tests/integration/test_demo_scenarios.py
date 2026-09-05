"""Integration test: demo agents -> public API -> trace context -> event
recording -> run lifecycle -> in-memory store, with no layer bypassed.
"""

from collections import Counter

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import SCENARIOS


def test_all_five_scenarios_end_to_end():
    lens = AgentLens()

    run_ids = {name: scenario(lens) for name, scenario in SCENARIOS.items()}

    # 1-2. all five runs exist in the single instance
    assert len(run_ids) == 5
    stored = {run.id: run for run in lens.list_runs()}
    assert set(run_ids.values()) == set(stored)

    # 3. all runs are SUCCESS with a finish time and a well-formed lifecycle
    for run_id in run_ids.values():
        run = lens.get_run(run_id)
        assert run.status is RunStatus.SUCCESS
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at

    # 4. each run has independent, non-leaking events
    all_event_ids: list = []
    for run_id in run_ids.values():
        events = lens.get_events(run_id)
        assert {e.run_id for e in events} == {run_id}
        assert [e.sequence_number for e in events] == list(range(len(events)))
        assert events[0].event_type is EventType.RUN_STARTED
        assert events[-1].event_type is EventType.RUN_COMPLETED
        all_event_ids.extend(e.id for e in events)
    assert len(all_event_ids) == len(set(all_event_ids))

    # 5. the expected problematic pattern shows up in the right scenario only

    def types(name: str) -> Counter:
        return Counter(e.event_type for e in lens.get_events(run_ids[name]))

    def decision_names(name: str) -> list[str]:
        return [
            e.name for e in lens.get_events(run_ids[name]) if e.event_type is EventType.DECISION
        ]

    # normal: no repeated decision, exactly one tool call, no errors
    assert types("normal")[EventType.ERROR] == 0
    assert types("normal")[EventType.TOOL_CALL_STARTED] == 1
    assert len(decision_names("normal")) == len(set(decision_names("normal")))

    # looping: analyze_request / search_for_information repeat >= 3 times
    assert decision_names("looping").count("analyze_request") >= 3
    assert decision_names("looping").count("search_for_information") >= 3

    # retrying: >= 3 ERROR events, but the run still succeeded
    assert types("retrying")[EventType.ERROR] >= 3
    assert types("retrying")[EventType.TOOL_CALL_COMPLETED] == 1

    # duplicate: the same tool call (name + input) appears twice
    dup_starts = [
        (e.name, tuple(sorted(e.input.items())))
        for e in lens.get_events(run_ids["duplicate"])
        if e.event_type is EventType.TOOL_CALL_STARTED
    ]
    assert len(dup_starts) == 2
    assert dup_starts[0] == dup_starts[1]

    # inefficient: broad retrieval present, no errors, no repeated decision/tool
    ineff_tools = [
        e.name
        for e in lens.get_events(run_ids["inefficient"])
        if e.event_type is EventType.TOOL_CALL_STARTED
    ]
    assert "fetch_all_customers" in ineff_tools
    assert types("inefficient")[EventType.ERROR] == 0
    assert len(ineff_tools) == len(set(ineff_tools))


def test_scenarios_are_isolated_across_separate_instances():
    lens_a = AgentLens()
    lens_b = AgentLens()

    a_ids = [scenario(lens_a) for scenario in SCENARIOS.values()]
    b_ids = [scenario(lens_b) for scenario in SCENARIOS.values()]

    for a_id in a_ids:
        assert lens_b.get_run(a_id) is None
    for b_id in b_ids:
        assert lens_a.get_run(b_id) is None

    assert len(lens_a.list_runs()) == 5
    assert len(lens_b.list_runs()) == 5

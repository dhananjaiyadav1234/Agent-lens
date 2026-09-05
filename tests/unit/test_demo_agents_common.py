"""Cross-scenario guarantees shared by all demo agents."""

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus
from examples.demo_agents import SCENARIOS


def test_registry_exposes_the_five_scenarios():
    assert set(SCENARIOS) == {"normal", "looping", "retrying", "duplicate", "inefficient"}


def test_all_scenarios_use_the_supplied_instance_and_stay_independent():
    lens = AgentLens()
    run_ids = {name: scenario(lens) for name, scenario in SCENARIOS.items()}

    # every run lives in the one supplied instance
    assert {r.id for r in lens.list_runs()} == set(run_ids.values())
    assert len(lens.list_completed_runs()) == len(SCENARIOS)

    # ids are all distinct
    assert len(set(run_ids.values())) == len(SCENARIOS)


def test_events_do_not_leak_between_runs():
    lens = AgentLens()
    run_ids = {name: scenario(lens) for name, scenario in SCENARIOS.items()}

    for run_id in run_ids.values():
        events = lens.get_events(run_id)
        assert events, "scenario produced no events"
        assert {e.run_id for e in events} == {run_id}


def test_all_runs_are_retrievable_and_successful():
    lens = AgentLens()
    run_ids = [scenario(lens) for scenario in SCENARIOS.values()]
    for run_id in run_ids:
        run = lens.get_run(run_id)
        assert run is not None
        assert run.status is RunStatus.SUCCESS


def test_every_scenario_starts_and_ends_with_the_lifecycle_events():
    lens = AgentLens()
    for scenario in SCENARIOS.values():
        events = lens.get_events(scenario(lens))
        assert events[0].event_type is EventType.RUN_STARTED
        assert events[0].sequence_number == 0
        assert events[-1].event_type is EventType.RUN_COMPLETED


def test_running_scenarios_twice_produces_the_same_event_name_pattern():
    lens_a = AgentLens()
    lens_b = AgentLens()
    for name, scenario in SCENARIOS.items():
        pattern_a = [(e.event_type, e.name) for e in lens_a.get_events(scenario(lens_a))]
        pattern_b = [(e.event_type, e.name) for e in lens_b.get_events(scenario(lens_b))]
        assert pattern_a == pattern_b, f"scenario {name!r} is not deterministic"


def test_scenarios_do_not_create_their_own_lens():
    lens = AgentLens()
    for scenario in SCENARIOS.values():
        before = len(lens.list_runs())
        scenario(lens)
        assert len(lens.list_runs()) == before + 1

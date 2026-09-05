"""Integration: ``AgentLens.detect`` persists issues; ``get_issues`` reads them back."""

import pytest

from agentlens import AgentLens
from agentlens.models import RunStatus
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import SCENARIOS, run_looping_agent, run_normal_agent


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _content(issues) -> list:
    return [i.model_dump(mode="json", exclude={"id"}) for i in issues]


# ---------------------------------------------------------------------------
# D. detect() persists; get_issues() reads back (both stores)
# ---------------------------------------------------------------------------


def test_in_memory_detect_persists_issues():
    lens = AgentLens()
    run_id = run_looping_agent(lens)
    issues = lens.detect(run_id)
    assert issues  # looping produces at least one
    assert lens.get_issues(run_id) == issues


def test_sqlite_detect_persists_issues(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        run_id = run_looping_agent(lens)
        issues = lens.detect(run_id)
        assert lens.get_issues(run_id) == issues
    finally:
        store.close()


def test_get_issues_does_not_run_detection(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        run_id = run_looping_agent(lens)
        assert lens.get_issues(run_id) == []  # nothing detected yet
        lens.detect(run_id)
        assert len(lens.get_issues(run_id)) >= 1
    finally:
        store.close()


def test_get_issues_unknown_run_is_empty():
    lens = AgentLens()
    from uuid import uuid4

    assert lens.get_issues(uuid4()) == []


# ---------------------------------------------------------------------------
# E. Repeated detection appends batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_sqlite", [False, True])
def test_repeated_detection_appends_and_stays_deterministic(use_sqlite, tmp_path):
    store = SQLiteTraceStore(tmp_path / "r.db") if use_sqlite else None
    try:
        lens = AgentLens(store=store) if store else AgentLens()
        run_id = run_looping_agent(lens)
        run_before = lens.get_run(run_id).model_dump(mode="json")
        events_before = [e.model_dump(mode="json") for e in lens.get_events(run_id)]

        first = lens.detect(run_id)
        second = lens.detect(run_id)

        # detector content (minus fresh ids) is deterministic between batches
        assert _content(first) == _content(second)
        # ids actually differ
        assert {i.id for i in first}.isdisjoint({i.id for i in second})

        stored = lens.get_issues(run_id)
        assert len(stored) == len(first) + len(second)
        assert stored[: len(first)] == first  # batch 1 then batch 2, in order
        assert stored[len(first) :] == second

        # run / events untouched by detection + persistence
        assert lens.get_run(run_id).model_dump(mode="json") == run_before
        assert [e.model_dump(mode="json") for e in lens.get_events(run_id)] == events_before
        assert lens.get_run(run_id).status is RunStatus.SUCCESS
    finally:
        if store:
            store.close()


# ---------------------------------------------------------------------------
# F. Reopen
# ---------------------------------------------------------------------------


def test_issues_survive_close_and_reopen_then_detect_again(db_path):
    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    run_id = run_looping_agent(lens_a)
    generated = lens_a.detect(run_id)
    generated_dump = [i.model_dump(mode="json") for i in generated]
    run_dump = lens_a.get_run(run_id).model_dump(mode="json")
    events_dump = [e.model_dump(mode="json") for e in lens_a.get_events(run_id)]
    store_a.close()
    del lens_a, store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        lens_b = AgentLens(store=store_b)
        assert lens_b.get_run(run_id).model_dump(mode="json") == run_dump
        assert [e.model_dump(mode="json") for e in lens_b.get_events(run_id)] == events_dump

        restored = lens_b.get_issues(run_id)
        assert restored == generated
        assert [i.model_dump(mode="json") for i in restored] == generated_dump

        # detecting again after reopen appends a further batch
        again = lens_b.detect(run_id)
        assert _content(again) == _content(generated)
        assert len(lens_b.get_issues(run_id)) == len(generated) + len(again)
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# G. All demo scenarios round-trip through SQLite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_demo_scenario_persisted_issues_match_detection_output(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    run_id = scenario(lens_a)
    detected = [i.model_dump(mode="json") for i in lens_a.detect(run_id)]
    store_a.close()
    del lens_a, store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        lens_b = AgentLens(store=store_b)
        persisted = [i.model_dump(mode="json") for i in lens_b.get_issues(run_id)]
        assert persisted == detected
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_issues_from_one_run_never_appear_in_another(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        loop_id = run_looping_agent(lens)
        normal_id = run_normal_agent(lens)

        lens.detect(loop_id)
        lens.detect(normal_id)

        loop_issues = lens.get_issues(loop_id)
        assert loop_issues
        assert lens.get_issues(normal_id) == []
        assert all(i.run_id == loop_id for i in loop_issues)
        assert loop_id not in {i.run_id for i in lens.get_issues(normal_id)}
    finally:
        store.close()


def test_default_agentlens_is_still_in_memory():
    from agentlens.core.storage import InMemoryTraceStore

    assert isinstance(AgentLens()._store, InMemoryTraceStore)

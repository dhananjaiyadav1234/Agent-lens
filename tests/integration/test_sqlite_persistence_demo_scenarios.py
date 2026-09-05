"""Integration: ``AgentLens`` with a ``SQLiteTraceStore`` over the demo scenarios.

Covers the full path -- demo agent -> public tracing API -> SQLite -> reopen ->
``lens.detect`` -- and checks equivalence against the default in-memory store.
"""

import pytest

from agentlens import AgentLens
from agentlens.core.storage import InMemoryTraceStore
from agentlens.models import RunStatus
from agentlens.storage import SQLiteTraceStore
from examples.demo_agents import (
    SCENARIOS,
    run_looping_agent,
    run_normal_agent,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agentlens.db"


def _issue_shape(lens: AgentLens, run_id) -> list:
    """Deterministic issue content that is comparable across different runs.

    Excludes ``issue.id``, ``run_id`` and the raw ``related_event_ids`` (all
    freshly generated UUIDs), but resolves the related events to their
    ``(event_type, name, sequence_number)`` shape so structure is still checked.
    """

    events = {e.id: e for e in lens.get_events(run_id)}
    shapes = []
    for issue in lens.detect(run_id):
        related = [
            (events[eid].event_type.value, events[eid].name, events[eid].sequence_number)
            for eid in issue.related_event_ids
        ]
        shapes.append(
            (
                issue.issue_type.value,
                issue.severity.value,
                issue.description,
                issue.metadata,
                related,
            )
        )
    return shapes


def _full_issue_content(lens: AgentLens, run_id) -> list:
    """Full issue content (including run_id / related ids) minus the fresh issue id.

    Valid only when comparing detection of *the same* persisted run.
    """

    return [i.model_dump(mode="json", exclude={"id"}) for i in lens.detect(run_id)]


# ---------------------------------------------------------------------------
# In-memory default is unchanged
# ---------------------------------------------------------------------------


def test_agentlens_default_is_in_memory():
    lens = AgentLens()
    assert isinstance(lens._store, InMemoryTraceStore)


# ---------------------------------------------------------------------------
# E/F. Cross-store equivalence + detection after persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_sqlite_backed_scenario_matches_in_memory(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    mem_lens = AgentLens()
    mem_shape = _issue_shape(mem_lens, scenario(mem_lens))

    sqlite_store = SQLiteTraceStore(db_path)
    try:
        sqlite_lens = AgentLens(store=sqlite_store)
        sqlite_shape = _issue_shape(sqlite_lens, scenario(sqlite_lens))
    finally:
        sqlite_store.close()

    assert sqlite_shape == mem_shape


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_detection_is_identical_after_closing_and_reopening_the_database(scenario_name, db_path):
    scenario = SCENARIOS[scenario_name]

    store_a = SQLiteTraceStore(db_path)
    lens_a = AgentLens(store=store_a)
    run_id = scenario(lens_a)
    before = _full_issue_content(lens_a, run_id)
    run_before = lens_a.get_run(run_id).model_dump(mode="json")
    events_before = [e.model_dump(mode="json") for e in lens_a.get_events(run_id)]
    store_a.close()
    del lens_a, store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        lens_b = AgentLens(store=store_b)
        assert lens_b.get_run(run_id).model_dump(mode="json") == run_before
        assert [e.model_dump(mode="json") for e in lens_b.get_events(run_id)] == events_before
        assert _full_issue_content(lens_b, run_id) == before
    finally:
        store_b.close()


def test_run_stays_success_and_events_unchanged_after_persisted_detection(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        run_id = run_looping_agent(lens)
        events_before = [e.model_dump(mode="json") for e in lens.get_events(run_id)]

        first = _full_issue_content(lens, run_id)
        second = _full_issue_content(lens, run_id)  # G. repeated detection purity
        third = _full_issue_content(lens, run_id)

        assert first == second == third
        assert lens.get_run(run_id).status is RunStatus.SUCCESS
        assert [e.model_dump(mode="json") for e in lens.get_events(run_id)] == events_before
    finally:
        store.close()


# ---------------------------------------------------------------------------
# H. Isolation
# ---------------------------------------------------------------------------


def test_detecting_one_run_never_reads_another_runs_events(db_path):
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        loop_id = run_looping_agent(lens)
        normal_id = run_normal_agent(lens)

        loop_issues = lens.detect(loop_id)
        assert lens.detect(normal_id) == []  # normal stays clean despite loop run in same db

        loop_event_ids = {e.id for e in lens.get_events(loop_id)}
        normal_event_ids = {e.id for e in lens.get_events(normal_id)}
        assert loop_issues
        for issue in loop_issues:
            assert set(issue.related_event_ids) <= loop_event_ids
            assert set(issue.related_event_ids).isdisjoint(normal_event_ids)
    finally:
        store.close()


def test_separate_database_files_do_not_leak(tmp_path):
    store_a = SQLiteTraceStore(tmp_path / "a.db")
    store_b = SQLiteTraceStore(tmp_path / "b.db")
    try:
        lens_a = AgentLens(store=store_a)
        lens_b = AgentLens(store=store_b)
        a_id = run_looping_agent(lens_a)
        b_id = run_normal_agent(lens_b)

        assert lens_b.get_run(a_id) is None
        assert lens_a.get_run(b_id) is None
        assert len(lens_a.list_runs()) == 1
        assert len(lens_b.list_runs()) == 1
    finally:
        store_a.close()
        store_b.close()


# ---------------------------------------------------------------------------
# I. Serialization
# ---------------------------------------------------------------------------


def test_persisted_run_events_and_issues_json_round_trip(db_path):
    from agentlens.models import AgentEvent, AgentIssue, AgentRun

    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)
        run_id = run_looping_agent(lens)

        run = lens.get_run(run_id)
        assert AgentRun.model_validate_json(run.model_dump_json()) == run
        for event in lens.get_events(run_id):
            assert AgentEvent.model_validate_json(event.model_dump_json()) == event
        for issue in lens.detect(run_id):
            assert AgentIssue.model_validate_json(issue.model_dump_json()) == issue
    finally:
        store.close()

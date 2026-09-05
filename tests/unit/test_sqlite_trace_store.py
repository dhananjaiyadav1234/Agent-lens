"""Unit tests for :class:`agentlens.storage.SQLiteTraceStore`."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from agentlens.core.storage import InMemoryTraceStore, TraceStore
from agentlens.models import AgentEvent, AgentRun, EventType, RunStatus
from agentlens.storage import SQLiteTraceStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "trace.db"


@pytest.fixture
def store(db_path):
    s = SQLiteTraceStore(db_path)
    try:
        yield s
    finally:
        s.close()


_STARTED = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _run(**overrides) -> AgentRun:
    base = {"task": "t"}
    base.update(overrides)
    return AgentRun(**base)


def _event(run_id, seq, **overrides) -> AgentEvent:
    base = {
        "run_id": run_id,
        "sequence_number": seq,
        "event_type": EventType.DECISION,
        "name": f"step-{seq}",
    }
    base.update(overrides)
    return AgentEvent(**base)


# ---------------------------------------------------------------------------
# A. Store basics
# ---------------------------------------------------------------------------


def test_conforms_to_the_trace_store_protocol(store):
    assert isinstance(store, TraceStore)


def test_schema_is_created_automatically(db_path):
    SQLiteTraceStore(db_path).close()
    assert db_path.exists()
    # a second store over the same file must not fail on CREATE
    SQLiteTraceStore(db_path).close()


def test_save_and_get_run(store):
    run = _run()
    store.save_run(run)
    assert store.get_run(run.id) == run


def test_unknown_run_returns_none_like_in_memory(store):
    unknown = uuid4()
    assert store.get_run(unknown) is None
    assert InMemoryTraceStore().get_run(unknown) is None
    assert store.get_events(unknown) == []


def test_add_and_get_events(store):
    run = _run()
    store.save_run(run)
    e0 = _event(run.id, 0)
    e1 = _event(run.id, 1)
    store.add_event(e0)
    store.add_event(e1)
    assert store.get_events(run.id) == [e0, e1]


def test_events_returned_in_sequence_number_order_not_insertion_order(store):
    run = _run()
    store.save_run(run)
    for seq in (2, 0, 4, 1, 3):
        store.add_event(_event(run.id, seq))
    assert [e.sequence_number for e in store.get_events(run.id)] == [0, 1, 2, 3, 4]


def test_runs_are_isolated(store):
    a, b = _run(task="a"), _run(task="b")
    store.save_run(a)
    store.save_run(b)
    assert store.get_run(a.id).task == "a"
    assert store.get_run(b.id).task == "b"


def test_events_are_isolated_by_run_id(store):
    a, b = _run(), _run()
    store.save_run(a)
    store.save_run(b)
    store.add_event(_event(a.id, 0, name="a-only"))
    store.add_event(_event(b.id, 0, name="b-only"))
    assert [e.name for e in store.get_events(a.id)] == ["a-only"]
    assert [e.name for e in store.get_events(b.id)] == ["b-only"]


def test_list_runs_in_first_save_order_even_after_updates(store):
    a, b, c = _run(task="a"), _run(task="b"), _run(task="c")
    store.save_run(a)
    store.save_run(b)
    store.save_run(c)
    # update a -> must not move it to the end
    store.save_run(
        AgentRun(
            id=a.id,
            task="a",
            status=RunStatus.SUCCESS,
            started_at=a.started_at,
            finished_at=a.started_at + timedelta(seconds=1),
        )
    )
    assert [r.task for r in store.list_runs()] == ["a", "b", "c"]


def test_list_completed_runs(store):
    running = _run()
    done = AgentRun(
        task="done",
        status=RunStatus.SUCCESS,
        started_at=_STARTED,
        finished_at=_STARTED + timedelta(seconds=2),
    )
    failed = AgentRun(
        task="failed",
        status=RunStatus.FAILED,
        started_at=_STARTED,
        finished_at=_STARTED + timedelta(seconds=1),
    )
    store.save_run(running)
    store.save_run(done)
    store.save_run(failed)
    completed = {r.task for r in store.list_completed_runs()}
    assert completed == {"done", "failed"}


# ---------------------------------------------------------------------------
# B. Model round trips
# ---------------------------------------------------------------------------


def test_run_round_trip_preserves_all_fields(store):
    run = AgentRun(
        id=uuid4(),
        task="round trip",
        status=RunStatus.SUCCESS,
        started_at=datetime(2026, 3, 4, 5, 6, 7, 891011, tzinfo=UTC),
        finished_at=datetime(2026, 3, 4, 5, 6, 12, 0, tzinfo=UTC),
        metadata={"nested": {"list": [1, 2.5, "x", None, True], "flag": False}, "count": 3},
    )
    store.save_run(run)
    restored = store.get_run(run.id)
    assert restored == run
    assert restored.model_dump() == run.model_dump()
    assert restored.started_at == run.started_at
    assert restored.finished_at == run.finished_at


def test_event_round_trip_preserves_all_fields(store):
    run = _run()
    store.save_run(run)
    event = AgentEvent(
        id=uuid4(),
        run_id=run.id,
        sequence_number=7,
        timestamp=datetime(2026, 3, 4, 5, 6, 7, 42, tzinfo=UTC),
        event_type=EventType.TOOL_CALL_COMPLETED,
        name="lookup",
        input={"a": 1, "b": [1, 2, {"deep": None}]},
        output={"found": True, "rows": []},
        duration_ms=12.5,
        status="ok",
        metadata={"attempt": 2, "tags": ["x", "y"]},
    )
    store.add_event(event)
    (restored,) = store.get_events(run.id)
    assert restored == event
    assert restored.model_dump() == event.model_dump()


@pytest.mark.parametrize(
    ("field_input", "field_output"),
    [
        (None, None),
        ("a string", 42),
        (3.14, True),
        ([1, 2, 3], {"k": "v"}),
        ({"a": {"b": {"c": [None, False, 0]}}}, ["nested", ["list"]]),
    ],
)
def test_event_json_scalar_and_container_values_round_trip(store, field_input, field_output):
    run = _run()
    store.save_run(run)
    event = _event(
        run.id, 0, event_type=EventType.LLM_CALL_COMPLETED, input=field_input, output=field_output
    )
    store.add_event(event)
    (restored,) = store.get_events(run.id)
    assert restored.input == field_input
    assert restored.output == field_output
    assert restored == event


def test_running_run_with_no_finished_at_round_trips(store):
    run = _run()  # RUNNING, finished_at None
    store.save_run(run)
    restored = store.get_run(run.id)
    assert restored.status is RunStatus.RUNNING
    assert restored.finished_at is None


# ---------------------------------------------------------------------------
# reopen behaviour (critical)
# ---------------------------------------------------------------------------


def test_data_survives_closing_and_reopening_the_store(db_path):
    store_a = SQLiteTraceStore(db_path)
    run = _run(task="persist me")
    store_a.save_run(run)
    store_a.add_event(_event(run.id, 0, name="first"))
    store_a.add_event(_event(run.id, 1, name="second"))
    store_a.close()
    del store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        assert store_b.get_run(run.id) == run
        assert [e.name for e in store_b.get_events(run.id)] == ["first", "second"]
    finally:
        store_b.close()


def test_completed_run_state_survives_reopen(db_path):
    store_a = SQLiteTraceStore(db_path)
    run = _run()
    store_a.save_run(run)
    completed = AgentRun(
        id=run.id,
        task=run.task,
        status=RunStatus.SUCCESS,
        started_at=run.started_at,
        finished_at=run.started_at + timedelta(seconds=3),
        metadata={"k": "v"},
    )
    store_a.save_run(completed)
    store_a.close()

    store_b = SQLiteTraceStore(db_path)
    try:
        restored = store_b.get_run(run.id)
        assert restored == completed
        assert restored.status is RunStatus.SUCCESS
        assert restored.finished_at == run.started_at + timedelta(seconds=3)
        assert [r.id for r in store_b.list_completed_runs()] == [run.id]
    finally:
        store_b.close()


def test_separate_database_files_are_isolated(tmp_path):
    a = SQLiteTraceStore(tmp_path / "a.db")
    b = SQLiteTraceStore(tmp_path / "b.db")
    run = _run()
    a.save_run(run)
    try:
        assert a.get_run(run.id) == run
        assert b.get_run(run.id) is None
        assert b.list_runs() == []
    finally:
        a.close()
        b.close()


def test_close_is_idempotent(db_path):
    store = SQLiteTraceStore(db_path)
    store.close()
    store.close()  # must not raise


def test_context_manager_closes_the_store(db_path):
    with SQLiteTraceStore(db_path) as store:
        store.save_run(_run())
    with pytest.raises(Exception):  # noqa: B017 - sqlite3.ProgrammingError on a closed connection
        store.list_runs()


# ---------------------------------------------------------------------------
# cross-store equivalence
# ---------------------------------------------------------------------------


def test_matches_in_memory_store_for_the_same_operations(store):
    mem = InMemoryTraceStore()
    run = _run(task="compare")
    events = [_event(run.id, seq) for seq in (0, 2, 1)]
    for s in (mem, store):
        s.save_run(run)
        for event in events:
            s.add_event(event)

    assert store.get_run(run.id) == mem.get_run(run.id)
    assert store.get_events(run.id) == mem.get_events(run.id)
    assert [e.sequence_number for e in store.get_events(run.id)] == [
        e.sequence_number for e in mem.get_events(run.id)
    ]
    assert [r.id for r in store.list_runs()] == [r.id for r in mem.list_runs()]

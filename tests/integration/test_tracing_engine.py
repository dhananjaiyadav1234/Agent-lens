"""Integration tests for the tracing engine: end-to-end context behaviour,
instance isolation, and in-memory retrieval.
"""

from uuid import uuid4

import pytest

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus


def test_realistic_successful_trace():
    lens = AgentLens()

    with lens.trace("Find customer order") as trace:
        trace.record_event(
            event_type=EventType.DECISION,
            name="select_tool",
            input={"customer_id": "123"},
            output={"tool": "database_lookup"},
        )
        trace.record_event(
            event_type=EventType.TOOL_CALL_STARTED,
            name="database_lookup",
            input={"query": "SELECT * FROM orders WHERE customer_id = 123"},
        )
        trace.record_event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name="database_lookup",
            output={"orders": [{"id": "o-1"}]},
            duration_ms=42.0,
            status="ok",
        )

    run = lens.get_run(trace.run_id)
    events = lens.get_events(trace.run_id)

    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None
    assert [e.event_type for e in events] == [
        EventType.RUN_STARTED,
        EventType.DECISION,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.RUN_COMPLETED,
    ]
    assert [e.sequence_number for e in events] == [0, 1, 2, 3, 4]
    assert run in lens.list_completed_runs()


def test_realistic_failed_trace_reraises_and_records_error():
    lens = AgentLens()

    with pytest.raises(ValueError, match="no such customer"):
        with lens.trace("Find customer order") as trace:
            trace.record_event(event_type=EventType.DECISION, name="select_tool")
            raise ValueError("no such customer")

    run = lens.get_run(trace.run_id)
    events = lens.get_events(trace.run_id)
    assert run.status is RunStatus.FAILED
    assert events[-1].event_type is EventType.ERROR
    assert events[-1].output == {"exception_type": "ValueError", "message": "no such customer"}


def test_two_instances_do_not_interfere():
    lens_a = AgentLens()
    lens_b = AgentLens()

    with lens_a.trace("a-task") as ta:
        ta.record_event(event_type=EventType.DECISION, name="a-only")
        # b can run its own trace concurrently with a's context open
        with lens_b.trace("b-task") as tb:
            tb.record_event(event_type=EventType.DECISION, name="b-only")

    assert lens_a.get_run(tb.run_id) is None
    assert lens_b.get_run(ta.run_id) is None

    a_names = [e.name for e in lens_a.get_events(ta.run_id)]
    b_names = [e.name for e in lens_b.get_events(tb.run_id)]
    assert "b-only" not in a_names
    assert "a-only" not in b_names
    assert len(lens_a.list_runs()) == 1
    assert len(lens_b.list_runs()) == 1


def test_active_trace_state_is_independent_per_instance():
    lens_a = AgentLens()
    lens_b = AgentLens()
    with lens_a.trace("a") as ta:
        assert lens_a.active_trace is ta
        assert lens_b.active_trace is None
    assert lens_a.active_trace is None


def test_in_memory_retrieval_after_multiple_runs():
    lens = AgentLens()
    run_ids = []
    for i in range(3):
        with lens.trace(f"task-{i}") as trace:
            trace.record_event(event_type=EventType.DECISION, name=f"d-{i}")
        run_ids.append(trace.run_id)

    # retrieve run by id
    for i, rid in enumerate(run_ids):
        run = lens.get_run(rid)
        assert run is not None
        assert run.task == f"task-{i}"
        assert run.status is RunStatus.SUCCESS

    # events do not leak between runs
    for i, rid in enumerate(run_ids):
        names = [e.name for e in lens.get_events(rid)]
        assert names == ["run_started", f"d-{i}", "run_completed"]

    # unknown ids
    assert lens.get_run(uuid4()) is None
    assert lens.get_events(uuid4()) == []

    # completed runs remain accessible
    assert len(lens.list_completed_runs()) == 3


def test_completed_runs_remain_accessible_after_new_trace():
    lens = AgentLens()
    with lens.trace("first") as first:
        pass
    with lens.trace("second"):
        pass

    assert lens.get_run(first.run_id).status is RunStatus.SUCCESS
    assert len(lens.get_events(first.run_id)) == 2

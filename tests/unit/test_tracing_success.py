"""Successful-run behaviour of the tracing engine."""

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus


def test_run_starts_running_with_no_finish_time():
    lens = AgentLens()
    with lens.trace("do the thing") as trace:
        run = trace.run
        assert run.status is RunStatus.RUNNING
        assert run.finished_at is None
        assert run.task == "do the thing"


def test_run_started_is_the_first_event():
    lens = AgentLens()
    with lens.trace("t") as trace:
        events = trace.events()
        assert len(events) == 1
        assert events[0].event_type is EventType.RUN_STARTED
        assert events[0].sequence_number == 0


def test_events_can_be_recorded_and_are_ordered():
    lens = AgentLens()
    with lens.trace("t") as trace:
        trace.record_event(event_type=EventType.DECISION, name="a")
        trace.record_event(event_type=EventType.TOOL_CALL_STARTED, name="b")
        trace.record_event(event_type=EventType.TOOL_CALL_COMPLETED, name="c")

    events = lens.get_events(trace.run_id)
    names = [e.name for e in events]
    seqs = [e.sequence_number for e in events]
    assert names == ["run_started", "a", "b", "c", "run_completed"]
    assert seqs == [0, 1, 2, 3, 4]


def test_run_becomes_success_with_finished_at_and_completion_event():
    lens = AgentLens()
    with lens.trace("t") as trace:
        trace.record_event(event_type=EventType.DECISION, name="pick")

    run = lens.get_run(trace.run_id)
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at is not None
    assert run.finished_at >= run.started_at

    events = lens.get_events(trace.run_id)
    assert events[-1].event_type is EventType.RUN_COMPLETED
    # RUN_COMPLETED is recorded before the SUCCESS transition, so its timestamp
    # is at or before finished_at.
    assert events[-1].timestamp <= run.finished_at


def test_recorded_events_belong_to_the_run():
    lens = AgentLens()
    with lens.trace("t") as trace:
        trace.record_event(event_type=EventType.DECISION, name="x")

    run_id = trace.run_id
    assert all(e.run_id == run_id for e in lens.get_events(run_id))


def test_record_event_returns_the_event():
    lens = AgentLens()
    with lens.trace("t") as trace:
        event = trace.record_event(
            event_type=EventType.LLM_CALL_COMPLETED,
            name="model",
            input={"prompt": "hi"},
            output={"text": "hello"},
            duration_ms=12.5,
            status="ok",
            metadata={"tokens": 3},
        )
    assert event.sequence_number == 1
    assert event.input == {"prompt": "hi"}
    assert event.output == {"text": "hello"}
    assert event.duration_ms == 12.5
    assert event.status == "ok"
    assert event.metadata == {"tokens": 3}


def test_trace_is_closed_and_inactive_after_exit():
    lens = AgentLens()
    with lens.trace("t") as trace:
        assert lens.active_trace is trace
        assert not trace.closed
    assert trace.closed
    assert lens.active_trace is None


def test_lens_record_event_convenience_uses_active_trace():
    lens = AgentLens()
    with lens.trace("t") as trace:
        lens.record_event(event_type=EventType.DECISION, name="via_lens")
    assert [e.name for e in lens.get_events(trace.run_id)] == [
        "run_started",
        "via_lens",
        "run_completed",
    ]


def test_metadata_is_passed_through_to_the_run():
    lens = AgentLens()
    with lens.trace("t", metadata={"env": "test", "k": 1}) as trace:
        pass
    assert lens.get_run(trace.run_id).metadata == {"env": "test", "k": 1}

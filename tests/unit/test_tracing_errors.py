"""Invalid-operation error handling for the tracing engine."""

import pytest

from agentlens import AgentLens, AgentLensError, RunLifecycleError, TraceStateError
from agentlens.models import EventType


def test_record_event_without_active_trace_raises():
    lens = AgentLens()
    with pytest.raises(TraceStateError, match="no active trace"):
        lens.record_event(event_type=EventType.DECISION, name="x")


def test_record_event_after_successful_completion_raises():
    lens = AgentLens()
    with lens.trace("t") as trace:
        pass
    with pytest.raises(TraceStateError, match="already closed"):
        trace.record_event(event_type=EventType.DECISION, name="late")


def test_record_event_after_failure_raises():
    lens = AgentLens()
    try:
        with lens.trace("t") as trace:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    with pytest.raises(TraceStateError, match="already closed"):
        trace.record_event(event_type=EventType.DECISION, name="late")


def test_nested_trace_is_rejected():
    lens = AgentLens()
    with lens.trace("outer"):
        with pytest.raises(TraceStateError, match="nested traces are not supported"):
            with lens.trace("inner"):
                pass


def test_active_trace_restored_after_rejected_nested_trace():
    lens = AgentLens()
    with lens.trace("outer") as outer:
        with pytest.raises(TraceStateError):
            with lens.trace("inner"):
                pass
        # outer trace is untouched and still usable
        assert lens.active_trace is outer
        outer.record_event(event_type=EventType.DECISION, name="still_here")
    assert [e.name for e in lens.get_events(outer.run_id)] == [
        "run_started",
        "still_here",
        "run_completed",
    ]


def test_completing_a_run_twice_via_manager_raises():
    lens = AgentLens()
    with lens.trace("t") as trace:
        pass
    with pytest.raises(RunLifecycleError, match="already SUCCESS"):
        lens._run_manager.complete_run(trace.run_id)


def test_failing_a_completed_run_raises():
    lens = AgentLens()
    with lens.trace("t") as trace:
        pass
    with pytest.raises(RunLifecycleError, match="already SUCCESS"):
        lens._run_manager.fail_run(trace.run_id)


def test_completing_a_failed_run_raises():
    lens = AgentLens()
    try:
        with lens.trace("t") as trace:
            raise RuntimeError("x")
    except RuntimeError:
        pass
    with pytest.raises(RunLifecycleError, match="already FAILED"):
        lens._run_manager.complete_run(trace.run_id)


def test_failing_a_failed_run_raises():
    lens = AgentLens()
    try:
        with lens.trace("t") as trace:
            raise RuntimeError("x")
    except RuntimeError:
        pass
    with pytest.raises(RunLifecycleError, match="already FAILED"):
        lens._run_manager.fail_run(trace.run_id)


def test_error_types_share_a_common_base():
    assert issubclass(TraceStateError, AgentLensError)
    assert issubclass(RunLifecycleError, AgentLensError)

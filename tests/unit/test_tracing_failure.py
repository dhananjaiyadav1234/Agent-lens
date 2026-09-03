"""Failed-run behaviour of the tracing engine."""

import json

import pytest

from agentlens import AgentLens
from agentlens.models import EventType, RunStatus


class _CustomError(Exception):
    pass


def test_exception_is_reraised_unchanged():
    lens = AgentLens()
    with pytest.raises(_CustomError, match="boom"):
        with lens.trace("t"):
            raise _CustomError("boom")


def test_run_becomes_failed_with_finished_at():
    lens = AgentLens()
    try:
        with lens.trace("t") as trace:
            raise _CustomError("boom")
    except _CustomError:
        pass

    run = lens.get_run(trace.run_id)
    assert run.status is RunStatus.FAILED
    assert run.finished_at is not None
    assert run.finished_at >= run.started_at


def test_error_event_is_recorded_last_and_is_json_safe():
    lens = AgentLens()
    try:
        with lens.trace("t") as trace:
            trace.record_event(event_type=EventType.DECISION, name="pick")
            raise _CustomError("something failed")
    except _CustomError:
        pass

    events = lens.get_events(trace.run_id)
    assert [e.event_type for e in events] == [
        EventType.RUN_STARTED,
        EventType.DECISION,
        EventType.ERROR,
    ]
    error_event = events[-1]
    assert error_event.name == "_CustomError"
    assert error_event.status == "error"
    assert error_event.output == {
        "exception_type": "_CustomError",
        "message": "something failed",
    }
    # the stored payload is plain JSON, not a Python object
    assert json.loads(json.dumps(error_event.output)) == error_event.output


def test_no_run_completed_event_on_failure():
    lens = AgentLens()
    try:
        with lens.trace("t") as trace:
            raise _CustomError("x")
    except _CustomError:
        pass
    types = {e.event_type for e in lens.get_events(trace.run_id)}
    assert EventType.RUN_COMPLETED not in types


def test_raw_exception_object_is_not_stored_anywhere():
    lens = AgentLens()
    exc = _CustomError("keep me out")
    try:
        with lens.trace("t") as trace:
            raise exc
    except _CustomError:
        pass

    for event in lens.get_events(trace.run_id):
        dumped = event.model_dump(mode="json")
        assert dumped["output"] is None or isinstance(dumped["output"], dict)
        # round-trips through json => contains no exception objects
        json.dumps(dumped)


def test_active_trace_cleared_after_failure():
    lens = AgentLens()
    try:
        with lens.trace("t"):
            raise _CustomError("x")
    except _CustomError:
        pass
    assert lens.active_trace is None

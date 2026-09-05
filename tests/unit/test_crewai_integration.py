"""Unit tests for the AgentLens ↔ CrewAI event-bus listener.

Fully offline: activity is driven through CrewAI's real event bus
(``crewai_event_bus.emit`` with the SDK's own event classes). No crew, no
models, no API keys, no network.
"""

import datetime as dt
import json
import subprocess
import sys
from uuid import uuid4

import crewai.events.event_listener as _crewai_event_listener
import pytest
from crewai.events import (
    AgentExecutionErrorEvent,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    TaskFailedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

from agentlens import AgentLens, AgentLensError, TraceStateError
from agentlens.integrations.crewai import AgentLensCrewAIListener
from agentlens.models import EventType

_NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def _quiet_crewai_console():
    """Silence CrewAI's built-in console listener during tests."""

    formatter = _crewai_event_listener.event_listener.formatter
    previous = formatter.verbose
    formatter.verbose = False
    yield
    formatter.verbose = previous


def _emit(*events):
    for event in events:
        crewai_event_bus.emit("test", event)


def _tool_started(name="lookup_customer", args=None):
    return ToolUsageStartedEvent(tool_name=name, tool_args=args or {"customer_id": "123"})


def _tool_finished(name="lookup_customer", args=None, output="ok"):
    return ToolUsageFinishedEvent(
        tool_name=name,
        tool_args=args or {"customer_id": "123"},
        output=output,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _tool_error(name="lookup_customer", args=None, error="boom"):
    return ToolUsageErrorEvent(
        tool_name=name, tool_args=args or {"customer_id": "123"}, error=error
    )


# ---------------------------------------------------------------------------
# Public API / optional-dependency isolation
# ---------------------------------------------------------------------------


def test_listener_is_public_constructs_and_exposes_run_id():
    lens = AgentLens()
    with lens.trace("t") as trace:
        listener = AgentLensCrewAIListener(lens=lens, run_id=trace.run_id)
        try:
            assert isinstance(listener, AgentLensCrewAIListener)
            assert listener.run_id == trace.run_id
        finally:
            listener.detach()


def test_base_package_does_not_import_crewai():
    code = (
        "import agentlens, agentlens.integrations, agentlens.export, agentlens.cli.main, sys; "
        "bad=[m for m in sys.modules if m.split('.')[0] in ('crewai',)]; "
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_run_raises_agentlens_error():
    with pytest.raises(AgentLensError, match="no run with id"):
        AgentLensCrewAIListener(lens=AgentLens(), run_id=uuid4())


def test_activity_outside_an_active_agentlens_trace_is_ignored():
    lens = AgentLens()
    with lens.trace("t") as trace:
        listener = AgentLensCrewAIListener(lens=lens, run_id=trace.run_id)
    # trace closed -> handlers buffer nothing
    _emit(_tool_started("late_tool"), _tool_finished("late_tool"))
    assert listener.flush() == 0
    listener.detach()

    names = {e.name for e in lens.get_events(trace.run_id)}
    assert "late_tool" not in names


def test_flush_into_a_closed_trace_raises_trace_state_error():
    lens = AgentLens()
    with lens.trace("t") as trace:
        listener = AgentLensCrewAIListener(lens=lens, run_id=trace.run_id)
        # buffer deterministically by invoking the handlers synchronously
        listener._on_tool_started("src", _tool_started())
        listener._on_tool_finished("src", _tool_finished())
    with pytest.raises(TraceStateError):
        listener.flush()
    listener.detach()


# ---------------------------------------------------------------------------
# Tool activity
# ---------------------------------------------------------------------------


def test_tool_start_and_end_map_and_preserve_data():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                _tool_started("lookup_customer", {"customer_id": "123"}),
                _tool_finished("lookup_customer", {"customer_id": "123"}, {"orders": 2}),
            )

    events = lens.get_events(trace.run_id)
    kinds = [(e.event_type, e.name) for e in events]
    assert (EventType.TOOL_CALL_STARTED, "lookup_customer") in kinds
    assert (EventType.TOOL_CALL_COMPLETED, "lookup_customer") in kinds

    started = next(e for e in events if e.event_type is EventType.TOOL_CALL_STARTED)
    completed = next(e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED)
    assert started.input == {"customer_id": "123"}
    assert completed.input == {"customer_id": "123"}
    assert completed.output == {"orders": 2}
    assert completed.status == "ok"
    assert started.metadata["framework"] == "crewai"
    assert started.metadata["framework_event_type"] == "tool_usage_started"
    assert "framework_event_id" in started.metadata
    assert all(e.run_id == trace.run_id for e in events)


def test_tool_error_becomes_error_event_without_completion():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                _tool_started("fetch", {"id": "1"}),
                _tool_error("fetch", {"id": "1"}, "upstream 500"),
            )

    events = lens.get_events(trace.run_id)
    assert not [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]
    error = next(e for e in events if e.event_type is EventType.ERROR)
    assert error.status == "error"
    assert error.input == {"id": "1"}
    assert error.output == {
        "operation": "fetch",
        "exception_type": "ToolUsageError",
        "message": "upstream 500",
    }


def test_self_reported_tool_failure_is_recorded_as_error_not_completion():
    from crewai.tools.tool_failure import ToolFailure

    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            finished = ToolUsageFinishedEvent(
                tool_name="fetch",
                tool_args={"id": "1"},
                output="could not fetch",
                started_at=_NOW,
                finished_at=_NOW,
                failure=ToolFailure(message="channel not found"),
            )
            _emit(_tool_started("fetch", {"id": "1"}), finished)

    events = lens.get_events(trace.run_id)
    assert not [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]
    error = next(e for e in events if e.event_type is EventType.ERROR)
    assert error.status == "error"
    assert error.output == {
        "operation": "fetch",
        "exception_type": "ToolFailure",
        "message": "channel not found",
    }


def test_tool_events_preserve_emission_order():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            for i in range(3):
                _emit(
                    _tool_started(f"tool{i}", {"i": i}),
                    _tool_finished(f"tool{i}", {"i": i}, {"r": i}),
                )

    events = [e for e in lens.get_events(trace.run_id) if e.name.startswith("tool")]
    assert [e.name for e in events] == [
        "tool0",
        "tool0",
        "tool1",
        "tool1",
        "tool2",
        "tool2",
    ]
    seqs = [e.sequence_number for e in lens.get_events(trace.run_id)]
    assert seqs == sorted(seqs) == list(range(len(seqs)))


# ---------------------------------------------------------------------------
# Model activity
# ---------------------------------------------------------------------------


def test_llm_events_map_with_model_name():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                LLMCallStartedEvent(
                    call_id="c1", model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
                ),
                LLMCallCompletedEvent(
                    call_id="c1", model="gpt-4o-mini", response="hello", call_type="llm_call"
                ),
            )

    events = lens.get_events(trace.run_id)
    started = next(e for e in events if e.event_type is EventType.LLM_CALL_STARTED)
    completed = next(e for e in events if e.event_type is EventType.LLM_CALL_COMPLETED)
    assert started.name == "gpt-4o-mini"
    assert completed.name == "gpt-4o-mini"
    assert started.input == {"messages": [{"role": "user", "content": "hi"}]}
    assert completed.output == {"response": "hello"}
    assert completed.status == "ok"


def test_llm_event_without_model_falls_back_to_llm():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(LLMCallStartedEvent(call_id="c1"))

    started = next(
        e for e in lens.get_events(trace.run_id) if e.event_type is EventType.LLM_CALL_STARTED
    )
    assert started.name == "llm"


def test_llm_failure_becomes_error_event():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                LLMCallStartedEvent(call_id="c1", model="m"),
                LLMCallFailedEvent(call_id="c1", model="m", error="rate limited"),
            )

    error = next(e for e in lens.get_events(trace.run_id) if e.event_type is EventType.ERROR)
    assert error.output == {
        "operation": "m",
        "exception_type": "LLMCallError",
        "message": "rate limited",
    }
    assert error.status == "error"


# ---------------------------------------------------------------------------
# Agent / task failures
# ---------------------------------------------------------------------------


def test_task_failure_maps_to_error_with_error_type():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(TaskFailedEvent(error="task blew up", error_type=ValueError))

    error = next(e for e in lens.get_events(trace.run_id) if e.event_type is EventType.ERROR)
    assert error.name == "ValueError"
    assert error.output == {
        "operation": "task",
        "exception_type": "ValueError",
        "message": "task blew up",
    }


def test_agent_execution_error_maps_to_error():
    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                AgentExecutionErrorEvent.model_construct(
                    error="agent crashed", agent=None, task=None
                )
            )

    error = next(e for e in lens.get_events(trace.run_id) if e.event_type is EventType.ERROR)
    assert error.output == {
        "operation": "agent_execution",
        "exception_type": "AgentExecutionError",
        "message": "agent crashed",
    }


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_non_json_objects_degrade_to_strings_and_round_trip():
    class Weird:
        def __repr__(self):
            return "<Weird>"

    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                _tool_started("x", {"n": 1}),
                _tool_finished("x", {"n": 1}, {"obj": Weird(), "nested": {"list": [1, None, 2.5]}}),
            )

    completed = next(
        e for e in lens.get_events(trace.run_id) if e.event_type is EventType.TOOL_CALL_COMPLETED
    )
    assert completed.output == {"obj": "<Weird>", "nested": {"list": [1, None, 2.5]}}
    json.dumps([e.model_dump(mode="json") for e in lens.get_events(trace.run_id)])


# ---------------------------------------------------------------------------
# Isolation / no side effects
# ---------------------------------------------------------------------------


def test_two_listeners_two_runs_do_not_leak():
    lens = AgentLens()

    listener_a = None
    try:
        with lens.trace("run a") as ta:
            listener_a = AgentLensCrewAIListener(lens=lens, run_id=ta.run_id)
            _emit(_tool_started("a_tool", {}), _tool_finished("a_tool", {}))
            assert listener_a.flush() == 2

        # listener_a stays registered on the global bus; run b's events must not
        # reach run a
        with lens.trace("run b") as tb:
            listener_b = AgentLensCrewAIListener(lens=lens, run_id=tb.run_id)
            try:
                _emit(_tool_started("b_tool", {}), _tool_finished("b_tool", {}))
                assert listener_a.flush() == 0
                assert listener_b.flush() == 2
            finally:
                listener_b.detach()
    finally:
        if listener_a is not None:
            listener_a.detach()

    a_names = {e.name for e in lens.get_events(ta.run_id)}
    b_names = {e.name for e in lens.get_events(tb.run_id)}
    assert "a_tool" in a_names and "b_tool" not in a_names
    assert "b_tool" in b_names and "a_tool" not in b_names


def test_activity_never_runs_detectors(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("run_detectors must not be called by the listener")

    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom)

    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(
                LLMCallStartedEvent(call_id="c1", model="m"),
                LLMCallCompletedEvent(call_id="c1", model="m", response="x", call_type="llm_call"),
                _tool_started("tool", {}),
                _tool_finished("tool", {}),
            )

    assert len(lens.get_events(trace.run_id)) >= 4
    assert lens.get_issues(trace.run_id) == []


def test_activity_never_builds_reports(monkeypatch):
    def _boom(self, run_id):
        raise AssertionError("get_report must not be called by the listener")

    monkeypatch.setattr("agentlens.AgentLens.get_report", _boom)

    lens = AgentLens()
    with lens.trace("t") as trace:
        with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
            _emit(_tool_started("tool", {}), _tool_finished("tool", {}))

    assert len(lens.get_events(trace.run_id)) >= 2

"""Unit tests for the AgentLens ↔ OpenAI Agents SDK tracing processor.

Fully offline: span activity is driven through the SDK's real tracing machinery
(``agents.tracing.trace`` + ``function_span`` / ``generation_span`` /
``handoff_span`` + ``set_trace_processors``). No models, no API keys, no network.
"""

import json
import subprocess
import sys
from uuid import uuid4

import pytest
from agents.tracing import (
    function_span,
    generation_span,
    handoff_span,
    set_trace_processors,
)
from agents.tracing import trace as agents_trace

from agentlens import AgentLens, AgentLensError, TraceStateError
from agentlens.integrations.openai_agents import AgentLensOpenAITracer
from agentlens.models import EventType


@pytest.fixture(autouse=True)
def _reset_trace_processors():
    yield
    set_trace_processors([])


def _drive(lens, run_id, body):
    """Register a tracer for ``run_id`` and run ``body()`` inside an OpenAI trace."""

    tracer = AgentLensOpenAITracer(lens=lens, run_id=run_id)
    set_trace_processors([tracer])
    with agents_trace("workflow"):
        body()
    set_trace_processors([])
    return tracer


# ---------------------------------------------------------------------------
# Public API / optional-dependency isolation
# ---------------------------------------------------------------------------


def test_tracer_is_public_constructs_and_exposes_run_id():
    lens = AgentLens()
    with lens.trace("t") as trace:
        tracer = AgentLensOpenAITracer(lens=lens, run_id=trace.run_id)
        assert isinstance(tracer, AgentLensOpenAITracer)
        assert tracer.run_id == trace.run_id


def test_base_package_does_not_import_the_sdk():
    code = (
        "import agentlens, agentlens.integrations, agentlens.export, agentlens.cli.main, sys; "
        "bad=[m for m in sys.modules if m.split('.')[0] in ('agents','openai')]; "
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_run_raises_agentlens_error():
    with pytest.raises(AgentLensError, match="no run with id"):
        AgentLensOpenAITracer(lens=AgentLens(), run_id=uuid4())


def test_spans_outside_an_active_agentlens_trace_are_ignored():
    lens = AgentLens()
    with lens.trace("t") as trace:
        tracer = AgentLensOpenAITracer(lens=lens, run_id=trace.run_id)
    set_trace_processors([tracer])
    # the AgentLens trace is closed -> the OpenAI trace is never adopted
    with agents_trace("wf"):
        with function_span(name="late_tool", input="{}") as span:
            span.span_data.output = "x"
    set_trace_processors([])

    names = {e.name for e in lens.get_events(trace.run_id)}
    assert "late_tool" not in names


def test_record_without_active_trace_raises_trace_state_error():
    lens = AgentLens()
    with lens.trace("t") as trace:
        tracer = AgentLensOpenAITracer(lens=lens, run_id=trace.run_id)
    # simulate an adopted span arriving after the trace closed
    with pytest.raises(TraceStateError):
        tracer._record(EventType.TOOL_CALL_STARTED, "x", _FakeSpan("trace_x", "span_x"))


class _FakeSpan:
    def __init__(self, trace_id: str, span_id: str) -> None:
        self.trace_id = trace_id
        self.span_id = span_id


# ---------------------------------------------------------------------------
# Tool activity
# ---------------------------------------------------------------------------


def test_tool_start_and_end_map_and_preserve_data():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with function_span(name="lookup_customer", input='{"customer_id": "123"}') as span:
                span.span_data.output = {"id": "123", "orders": 2}

        _drive(lens, trace.run_id, body)

    events = lens.get_events(trace.run_id)
    kinds = [(e.event_type, e.name) for e in events]
    assert (EventType.TOOL_CALL_STARTED, "lookup_customer") in kinds
    assert (EventType.TOOL_CALL_COMPLETED, "lookup_customer") in kinds

    started = next(e for e in events if e.event_type is EventType.TOOL_CALL_STARTED)
    completed = next(e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED)
    assert started.input == {"customer_id": "123"}
    assert completed.input == {"customer_id": "123"}
    assert completed.output == {"id": "123", "orders": 2}
    assert completed.status == "ok"
    assert started.metadata["framework"] == "openai_agents"
    assert "framework_run_id" in started.metadata and "framework_span_id" in started.metadata
    assert all(e.run_id == trace.run_id for e in events)


def test_tool_error_becomes_error_event_without_completion():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with function_span(name="fetch", input='{"id": "1"}') as span:
                span.set_error({"message": "boom", "data": {"exception_type": "RuntimeError"}})

        _drive(lens, trace.run_id, body)

    events = lens.get_events(trace.run_id)
    assert not [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]
    error = next(e for e in events if e.event_type is EventType.ERROR)
    assert error.name == "RuntimeError"
    assert error.status == "error"
    assert error.input == {"id": "1"}
    assert error.output["operation"] == "fetch"
    assert error.output["exception_type"] == "RuntimeError"
    assert error.output["message"] == "boom"


def test_tool_events_preserve_order():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            for i in range(3):
                with function_span(name=f"tool{i}", input=json.dumps({"i": i})) as span:
                    span.span_data.output = {"r": i}

        _drive(lens, trace.run_id, body)

    events = [e for e in lens.get_events(trace.run_id) if e.name.startswith("tool")]
    assert [e.name for e in events] == ["tool0", "tool0", "tool1", "tool1", "tool2", "tool2"]
    seqs = [e.sequence_number for e in lens.get_events(trace.run_id)]
    assert seqs == sorted(seqs) == list(range(len(seqs)))


# ---------------------------------------------------------------------------
# Model activity
# ---------------------------------------------------------------------------


def test_generation_span_maps_to_llm_events_with_model_name():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with generation_span(model="gpt-4o-mini") as span:
                span.span_data.output = [{"role": "assistant", "content": "hello"}]

        _drive(lens, trace.run_id, body)

    events = lens.get_events(trace.run_id)
    started = next(e for e in events if e.event_type is EventType.LLM_CALL_STARTED)
    completed = next(e for e in events if e.event_type is EventType.LLM_CALL_COMPLETED)
    assert started.name == "gpt-4o-mini"
    assert completed.name == "gpt-4o-mini"
    assert completed.output == {"output": [{"role": "assistant", "content": "hello"}]}
    assert completed.status == "ok"


def test_generation_span_without_model_falls_back_to_llm():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with generation_span() as span:
                span.span_data.output = [{"role": "assistant", "content": "x"}]

        _drive(lens, trace.run_id, body)

    started = next(
        e for e in lens.get_events(trace.run_id) if e.event_type is EventType.LLM_CALL_STARTED
    )
    assert started.name == "llm"


def test_generation_span_error():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with generation_span(model="m") as span:
                span.set_error({"message": "rate limited", "data": {"type": "RateLimitError"}})

        _drive(lens, trace.run_id, body)

    error = next(e for e in lens.get_events(trace.run_id) if e.event_type is EventType.ERROR)
    assert error.name == "RateLimitError"
    assert error.output == {
        "operation": "m",
        "exception_type": "RateLimitError",
        "message": "rate limited",
    }


# ---------------------------------------------------------------------------
# Agent decisions (handoff)
# ---------------------------------------------------------------------------


def test_handoff_span_maps_to_decision():
    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with handoff_span(from_agent="triage", to_agent="billing"):
                pass

        _drive(lens, trace.run_id, body)

    events = lens.get_events(trace.run_id)
    decision = next(e for e in events if e.event_type is EventType.DECISION)
    assert decision.name == "handoff"
    assert decision.input == {"from_agent": "triage", "to_agent": "billing"}


def test_agent_span_is_not_mapped():
    """AgentSpanData wraps a whole agent run; mapping it would only add noise."""

    from agents.tracing import agent_span

    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with agent_span(name="assistant"):
                with function_span(name="tool", input="{}") as span:
                    span.span_data.output = "ok"

        _drive(lens, trace.run_id, body)

    names = {e.name for e in lens.get_events(trace.run_id)}
    assert "assistant" not in names
    assert "tool" in names


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_non_json_objects_degrade_to_strings_and_round_trip():
    class Weird:
        def __repr__(self):
            return "<Weird>"

    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with function_span(name="x", input=json.dumps({"n": 1})) as span:
                span.span_data.output = {"obj": Weird(), "nested": {"list": [1, None, True, 2.5]}}

        _drive(lens, trace.run_id, body)

    events = lens.get_events(trace.run_id)
    completed = next(e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED)
    assert completed.output == {"obj": "<Weird>", "nested": {"list": [1, None, True, 2.5]}}
    json.dumps([e.model_dump(mode="json") for e in events])  # survives a JSON round-trip


# ---------------------------------------------------------------------------
# Isolation / no side effects
# ---------------------------------------------------------------------------


def test_two_tracers_two_runs_do_not_leak():
    lens = AgentLens()

    with lens.trace("run a") as ta:
        ta_tracer = AgentLensOpenAITracer(lens=lens, run_id=ta.run_id)
        set_trace_processors([ta_tracer])
        with agents_trace("wf a"):
            with function_span(name="a_tool", input="{}") as span:
                span.span_data.output = "a"

    # a_tracer stays globally registered; run b's spans must not reach run a
    with lens.trace("run b") as tb:
        tb_tracer = AgentLensOpenAITracer(lens=lens, run_id=tb.run_id)
        set_trace_processors([ta_tracer, tb_tracer])
        with agents_trace("wf b"):
            with function_span(name="b_tool", input="{}") as span:
                span.span_data.output = "b"
    set_trace_processors([])

    a_names = {e.name for e in lens.get_events(ta.run_id)}
    b_names = {e.name for e in lens.get_events(tb.run_id)}
    assert "a_tool" in a_names and "b_tool" not in a_names
    assert "b_tool" in b_names and "a_tool" not in b_names


def test_activity_never_runs_detectors(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("run_detectors must not be called by the tracer")

    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom)

    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with generation_span(model="m") as span:
                span.span_data.output = [{"role": "assistant", "content": "hi"}]
            with function_span(name="tool", input="{}") as span:
                span.span_data.output = "ok"

        _drive(lens, trace.run_id, body)

    assert len(lens.get_events(trace.run_id)) >= 4
    assert lens.get_issues(trace.run_id) == []


def test_activity_never_builds_reports(monkeypatch):
    def _boom(self, run_id):
        raise AssertionError("get_report must not be called by the tracer")

    monkeypatch.setattr("agentlens.AgentLens.get_report", _boom)

    lens = AgentLens()
    with lens.trace("t") as trace:

        def body():
            with function_span(name="tool", input="{}") as span:
                span.span_data.output = "ok"

        _drive(lens, trace.run_id, body)

    assert len(lens.get_events(trace.run_id)) >= 2

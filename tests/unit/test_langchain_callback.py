"""Unit tests for the AgentLens ↔ LangChain callback handler.

Fully offline: LLM/tool activity is driven either by invoking real (local)
``langchain_core`` tools/runnables or by calling the documented callback hooks
directly with representative arguments. No API keys, no network.
"""

import subprocess
import sys
from uuid import uuid4

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.tools import tool

from agentlens import AgentLens, AgentLensError, TraceStateError
from agentlens.integrations.langchain import AgentLensCallbackHandler
from agentlens.models import EventType


@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer by id (deterministic, offline)."""

    return {"id": customer_id, "orders": 2}


# ---------------------------------------------------------------------------
# Public API / import boundary
# ---------------------------------------------------------------------------


def test_handler_is_public_and_constructs():
    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        assert isinstance(handler, AgentLensCallbackHandler)
        assert handler.run_id == trace.run_id


def test_base_package_does_not_import_langchain():
    code = (
        "import agentlens, agentlens.export, agentlens.cli.main, agentlens.integrations, sys; "
        "bad=[m for m in sys.modules if m.split('.')[0] in ('langchain','langchain_core')]; "
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_run_raises_agentlens_error():
    with pytest.raises(AgentLensError, match="no run with id"):
        AgentLensCallbackHandler(lens=AgentLens(), run_id=uuid4())


def test_callback_outside_active_trace_raises_trace_state_error():
    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
    # trace is closed now
    with pytest.raises(TraceStateError):
        handler.on_tool_start({"name": "x"}, "in", run_id=uuid4())


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


def test_tool_start_and_end_via_real_invoke():
    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        lookup_customer.invoke({"customer_id": "123"}, config={"callbacks": [handler]})

    events = lens.get_events(trace.run_id)
    kinds = [(e.event_type, e.name) for e in events]
    assert (EventType.TOOL_CALL_STARTED, "lookup_customer") in kinds
    assert (EventType.TOOL_CALL_COMPLETED, "lookup_customer") in kinds

    started = next(e for e in events if e.event_type is EventType.TOOL_CALL_STARTED)
    completed = next(e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED)
    assert started.input == {"customer_id": "123"}
    assert completed.input == {"customer_id": "123"}  # same logical operation
    assert completed.output == {"id": "123", "orders": 2}
    assert completed.status == "ok"
    assert all(e.run_id == trace.run_id for e in events)


def test_tool_error_records_error_event_with_operation():
    lens = AgentLens()
    rid = uuid4()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        handler.on_tool_start({"name": "fetch"}, "{'id': '1'}", run_id=rid, inputs={"id": "1"})
        handler.on_tool_error(RuntimeError("boom"), run_id=rid)

    events = lens.get_events(trace.run_id)
    error = next(e for e in events if e.event_type is EventType.ERROR)
    assert error.name == "RuntimeError"
    assert error.status == "error"
    assert error.input == {"id": "1"}
    assert error.output == {
        "operation": "fetch",
        "exception_type": "RuntimeError",
        "message": "boom",
    }


def test_tool_error_does_not_synthesize_a_completion():
    lens = AgentLens()
    rid = uuid4()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        handler.on_tool_start({"name": "fetch"}, "in", run_id=rid, inputs={"a": 1})
        handler.on_tool_error(ValueError("nope"), run_id=rid)

    events = lens.get_events(trace.run_id)
    assert not [e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED]


def test_tool_error_reraises_from_real_invoke():
    @tool
    def boom(x: int) -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        with pytest.raises(RuntimeError, match="kaboom"):
            boom.invoke({"x": 1}, config={"callbacks": [handler]})

    events = lens.get_events(trace.run_id)
    assert [e.event_type for e in events if e.event_type is EventType.ERROR]


def test_tool_events_preserve_callback_invocation_order():
    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        for i in range(3):
            rid = uuid4()
            handler.on_tool_start({"name": f"tool{i}"}, "in", run_id=rid, inputs={"i": i})
            handler.on_tool_end({"r": i}, run_id=rid)

    events = [e for e in lens.get_events(trace.run_id) if e.event_type is not EventType.RUN_STARTED]
    names = [e.name for e in events if e.name.startswith("tool")]
    assert names == ["tool0", "tool0", "tool1", "tool1", "tool2", "tool2"]
    seqs = [e.sequence_number for e in lens.get_events(trace.run_id)]
    assert seqs == sorted(seqs) == list(range(len(seqs)))


# ---------------------------------------------------------------------------
# LLM events
# ---------------------------------------------------------------------------


def test_llm_start_end_and_error():
    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        r1 = uuid4()
        handler.on_llm_start({"name": "m"}, ["say hi"], run_id=r1)
        handler.on_llm_end(LLMResult(generations=[[Generation(text="hi")]]), run_id=r1)
        r2 = uuid4()
        handler.on_llm_start({"name": "m"}, ["fail"], run_id=r2)
        handler.on_llm_error(RuntimeError("rate limit"), run_id=r2)

    events = lens.get_events(trace.run_id)
    kinds = [e.event_type for e in events]
    assert EventType.LLM_CALL_STARTED in kinds
    assert EventType.LLM_CALL_COMPLETED in kinds

    started = next(e for e in events if e.event_type is EventType.LLM_CALL_STARTED)
    assert started.input == {"prompts": ["say hi"]}
    completed = next(e for e in events if e.event_type is EventType.LLM_CALL_COMPLETED)
    assert completed.output == {"generations": ["hi"]}
    assert completed.status == "ok"
    error = next(e for e in events if e.event_type is EventType.ERROR)
    assert error.output == {
        "operation": "m",
        "exception_type": "RuntimeError",
        "message": "rate limit",
    }


def test_chat_model_start_maps_to_llm_started():
    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        handler.on_chat_model_start({"name": "chat"}, [["hello"]], run_id=uuid4())
    started = [
        e for e in lens.get_events(trace.run_id) if e.event_type is EventType.LLM_CALL_STARTED
    ]
    assert len(started) == 1
    assert started[0].name == "chat"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_has_framework_marker_and_stringified_run_id():
    lens = AgentLens()
    rid = uuid4()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        handler.on_tool_start({"name": "x"}, "in", run_id=rid, inputs={"a": 1})
        handler.on_tool_end("done", run_id=rid)

    for event in lens.get_events(trace.run_id):
        if event.name == "x":
            assert event.metadata["framework"] == "langchain"
            assert event.metadata["langchain_run_id"] == str(rid)
            assert isinstance(event.metadata["langchain_run_id"], str)


def test_non_json_framework_objects_are_stringified_not_persisted_raw():
    class Weird:
        def __repr__(self) -> str:
            return "<Weird>"

    lens = AgentLens()
    rid = uuid4()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        handler.on_tool_start({"name": "x"}, "in", run_id=rid, inputs={"obj": Weird(), "n": 1})
        handler.on_tool_end(Weird(), run_id=rid)

    events = lens.get_events(trace.run_id)
    started = next(e for e in events if e.event_type is EventType.TOOL_CALL_STARTED)
    completed = next(e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED)
    assert started.input == {"obj": "<Weird>", "n": 1}
    assert completed.output == "<Weird>"
    # round-trips as plain JSON
    import json

    json.dumps([e.model_dump(mode="json") for e in events])


def test_nested_json_values_survive():
    lens = AgentLens()
    rid = uuid4()
    payload = {"a": [1, 2, {"b": None, "c": True}], "d": {"e": [False, "x"]}}
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        handler.on_tool_start({"name": "x"}, "in", run_id=rid, inputs=payload)
        handler.on_tool_end(payload, run_id=rid)
    events = lens.get_events(trace.run_id)
    assert next(e for e in events if e.event_type is EventType.TOOL_CALL_STARTED).input == payload
    assert (
        next(e for e in events if e.event_type is EventType.TOOL_CALL_COMPLETED).output == payload
    )


# ---------------------------------------------------------------------------
# Run isolation / no side effects
# ---------------------------------------------------------------------------


def test_two_handlers_two_runs_do_not_leak():
    lens = AgentLens()
    with lens.trace("run a") as ta:
        ha = AgentLensCallbackHandler(lens=lens, run_id=ta.run_id)
        r = uuid4()
        ha.on_tool_start({"name": "a_tool"}, "in", run_id=r, inputs={})
        ha.on_tool_end("a", run_id=r)
    with lens.trace("run b") as tb:
        hb = AgentLensCallbackHandler(lens=lens, run_id=tb.run_id)
        r = uuid4()
        hb.on_tool_start({"name": "b_tool"}, "in", run_id=r, inputs={})
        hb.on_tool_end("b", run_id=r)

    a_names = {e.name for e in lens.get_events(ta.run_id)}
    b_names = {e.name for e in lens.get_events(tb.run_id)}
    assert "a_tool" in a_names and "b_tool" not in a_names
    assert "b_tool" in b_names and "a_tool" not in b_names


def test_callback_activity_never_runs_detectors(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("run_detectors must not be called by the callback handler")

    monkeypatch.setattr("agentlens.core.lens.run_detectors", _boom)

    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        lookup_customer.invoke({"customer_id": "1"}, config={"callbacks": [handler]})
        r = uuid4()
        handler.on_llm_start({"name": "m"}, ["p"], run_id=r)
        handler.on_llm_end(LLMResult(generations=[[Generation(text="x")]]), run_id=r)

    assert len(lens.get_events(trace.run_id)) >= 4
    assert lens.get_issues(trace.run_id) == []


def test_callback_activity_never_builds_reports(monkeypatch):
    def _boom(self, run_id):
        raise AssertionError("get_report must not be called by the callback handler")

    monkeypatch.setattr("agentlens.AgentLens.get_report", _boom)

    lens = AgentLens()
    with lens.trace("t") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
        lookup_customer.invoke({"customer_id": "1"}, config={"callbacks": [handler]})
    assert len(lens.get_events(trace.run_id)) >= 3

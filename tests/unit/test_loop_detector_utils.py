"""Tests for the detector fingerprint / canonical-JSON helpers."""

from uuid import uuid4

from agentlens.detectors.utils import canonical_json, event_fingerprint
from agentlens.models import AgentEvent, EventType

_RUN_ID = uuid4()


def _event(name, *, event_type=EventType.DECISION, input=None, metadata=None):
    return AgentEvent(
        run_id=_RUN_ID,
        sequence_number=0,
        event_type=event_type,
        name=name,
        input=input,
        metadata=metadata or {},
    )


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_normalizes_nested_dicts():
    left = {"outer": {"a": 1, "b": {"x": 1, "y": 2}}}
    right = {"outer": {"b": {"y": 2, "x": 1}, "a": 1}}
    assert canonical_json(left) == canonical_json(right)


def test_canonical_json_preserves_list_order():
    assert canonical_json([1, 2, 3]) != canonical_json([3, 2, 1])


def test_canonical_json_keeps_primitive_distinctions():
    forms = {
        canonical_json(1),
        canonical_json(1.0),
        canonical_json("1"),
        canonical_json(True),
        canonical_json(None),
    }
    assert len(forms) == 5


def test_canonical_json_does_not_mutate_input():
    value = {"b": 2, "a": 1, "nested": {"z": 0, "y": 1}}
    snapshot = {"b": 2, "a": 1, "nested": {"z": 0, "y": 1}}
    canonical_json(value)
    assert value == snapshot
    assert list(value) == list(snapshot)


def test_fingerprint_ignores_metadata():
    a = _event("analyze_request", input={"q": 1}, metadata={"iteration": 0})
    b = _event("analyze_request", input={"q": 1}, metadata={"iteration": 7})
    assert event_fingerprint(a) == event_fingerprint(b)


def test_fingerprint_ignores_output_status_duration_and_ids():
    a = AgentEvent(
        run_id=uuid4(),
        sequence_number=1,
        event_type=EventType.TOOL_CALL_COMPLETED,
        name="lookup",
        input={"id": "1"},
        output={"found": True},
        status="ok",
        duration_ms=12.0,
    )
    b = AgentEvent(
        run_id=uuid4(),
        sequence_number=99,
        event_type=EventType.TOOL_CALL_COMPLETED,
        name="lookup",
        input={"id": "1"},
        output={"found": False, "extra": [1, 2]},
        status="error",
        duration_ms=999.0,
    )
    assert event_fingerprint(a) == event_fingerprint(b)


def test_fingerprint_distinguishes_type_name_and_input():
    base = _event("analyze_request", input={"q": 1})
    assert event_fingerprint(base) != event_fingerprint(_event("search", input={"q": 1}))
    assert event_fingerprint(base) != event_fingerprint(_event("analyze_request", input={"q": 2}))
    assert event_fingerprint(base) != event_fingerprint(
        _event("analyze_request", event_type=EventType.TOOL_CALL_STARTED, input={"q": 1})
    )


def test_fingerprint_equal_for_reordered_input_keys():
    a = _event("decide", input={"a": 1, "b": 2})
    b = _event("decide", input={"b": 2, "a": 1})
    assert event_fingerprint(a) == event_fingerprint(b)

"""Tests for :class:`agentlens.models.AgentEvent`."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentlens.models import AgentEvent, EventType


def _make(**overrides):
    base = {
        "run_id": uuid4(),
        "sequence_number": 0,
        "event_type": EventType.TOOL_CALL_STARTED,
        "name": "search",
    }
    base.update(overrides)
    return AgentEvent(**base)


def test_valid_creation_with_defaults():
    run_id = uuid4()
    event = AgentEvent(
        run_id=run_id,
        sequence_number=3,
        event_type=EventType.LLM_CALL_COMPLETED,
        name="gpt-like-model",
    )
    assert isinstance(event.id, UUID)
    assert event.run_id == run_id
    assert event.sequence_number == 3
    assert event.timestamp.tzinfo is not None
    assert event.input is None
    assert event.output is None
    assert event.duration_ms is None
    assert event.status is None
    assert event.metadata == {}


def test_negative_sequence_number_is_rejected():
    with pytest.raises(ValidationError):
        _make(sequence_number=-1)


def test_sequence_number_zero_is_allowed():
    assert _make(sequence_number=0).sequence_number == 0


def test_negative_duration_is_rejected():
    with pytest.raises(ValidationError):
        _make(duration_ms=-0.5)


def test_zero_and_positive_duration_allowed():
    assert _make(duration_ms=0).duration_ms == 0
    assert _make(duration_ms=1234.5).duration_ms == 1234.5


def test_run_id_accepts_uuid_compatible_string():
    raw = str(uuid4())
    assert _make(run_id=raw).run_id == UUID(raw)


def test_invalid_run_id_is_rejected():
    with pytest.raises(ValidationError):
        _make(run_id="nope")


def test_invalid_event_type_is_rejected():
    with pytest.raises(ValidationError):
        _make(event_type="teleport")


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        _make(timestamp=datetime(2026, 1, 1, 0, 0))


@pytest.mark.parametrize(
    "payload",
    [
        "a string",
        42,
        3.14,
        True,
        None,
        ["a", 1, False, None],
        {"nested": {"list": [1, 2, 3], "flag": True, "empty": None}},
    ],
)
def test_json_compatible_input_and_output(payload):
    event = _make(input=payload, output=payload)
    assert event.input == payload
    assert event.output == payload


def test_non_json_input_is_rejected():
    with pytest.raises(ValidationError):
        _make(input={"when": datetime(2026, 1, 1, tzinfo=UTC)})

    with pytest.raises(ValidationError):
        _make(output={1, 2, 3})


def test_metadata_must_be_json_object():
    event = _make(metadata={"retry": 2, "cached": False, "tags": ["a", "b"]})
    assert event.metadata["retry"] == 2

    with pytest.raises(ValidationError):
        _make(metadata={"obj": object()})


def test_status_is_free_form_string():
    assert _make(status="ok").status == "ok"
    assert _make(status="429").status == "429"


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        _make(extra_thing=1)

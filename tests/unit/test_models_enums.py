"""Tests for the trace-model enumerations."""

import pytest
from pydantic import BaseModel, ValidationError

from agentlens.models import EventType, IssueType, RunStatus, Severity


@pytest.mark.parametrize(
    ("enum_cls", "members"),
    [
        (RunStatus, {"RUNNING", "SUCCESS", "FAILED"}),
        (
            EventType,
            {
                "RUN_STARTED",
                "LLM_CALL_STARTED",
                "LLM_CALL_COMPLETED",
                "TOOL_CALL_STARTED",
                "TOOL_CALL_COMPLETED",
                "DECISION",
                "ERROR",
                "RUN_COMPLETED",
            },
        ),
        (
            IssueType,
            {
                "AGENT_LOOP",
                "EXCESSIVE_RETRY",
                "DUPLICATE_TOOL_CALL",
                "TOOL_FAILURE",
                "INEFFICIENCY",
            },
        ),
        (Severity, {"LOW", "MEDIUM", "HIGH", "CRITICAL"}),
    ],
)
def test_enum_has_exactly_expected_members(enum_cls, members):
    assert {m.name for m in enum_cls} == members


def test_enum_values_are_lowercase_strings():
    assert RunStatus.SUCCESS == "success"
    assert EventType.TOOL_CALL_STARTED == "tool_call_started"
    assert isinstance(Severity.HIGH, str)


def test_enum_round_trips_through_value():
    assert RunStatus("failed") is RunStatus.FAILED
    assert IssueType("agent_loop") is IssueType.AGENT_LOOP


def test_invalid_enum_value_raises():
    with pytest.raises(ValueError):
        RunStatus("not-a-status")


def test_invalid_enum_value_in_model_raises():
    class Wrapper(BaseModel):
        status: RunStatus

    with pytest.raises(ValidationError):
        Wrapper(status="bogus")

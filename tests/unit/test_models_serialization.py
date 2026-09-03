"""Round-trip serialization tests for the universal trace model."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agentlens.models import (
    AgentEvent,
    AgentIssue,
    AgentRun,
    EventType,
    IssueType,
    RunStatus,
    Severity,
)


def test_agent_run_json_round_trip():
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    run = AgentRun(
        task="do the thing",
        status=RunStatus.SUCCESS,
        started_at=started,
        finished_at=started + timedelta(seconds=2),
        metadata={"model": "sonnet", "cost_usd": 0.01, "tags": ["a", "b"]},
    )

    restored = AgentRun.model_validate_json(run.model_dump_json())
    assert restored == run


def test_agent_event_json_round_trip_preserves_payloads():
    event = AgentEvent(
        run_id=uuid4(),
        sequence_number=7,
        event_type=EventType.TOOL_CALL_COMPLETED,
        name="web.search",
        input={"query": "agentlens", "k": 5},
        output=["r1", "r2", {"score": 0.4, "ok": True, "note": None}],
        duration_ms=812.5,
        status="ok",
        metadata={"attempt": 1},
    )

    restored = AgentEvent.model_validate(event.model_dump(mode="json"))
    assert restored == event
    assert restored.input == event.input
    assert restored.output == event.output


def test_agent_issue_json_round_trip():
    issue = AgentIssue(
        run_id=uuid4(),
        issue_type=IssueType.DUPLICATE_TOOL_CALL,
        severity=Severity.MEDIUM,
        description="Identical tool call issued twice.",
        related_event_ids=[uuid4(), uuid4()],
        metadata={"detector": "dup-v1"},
    )

    restored = AgentIssue.model_validate_json(issue.model_dump_json())
    assert restored == issue


def test_dump_mode_json_is_pure_json_types():
    run = AgentRun(task="t")
    dumped = run.model_dump(mode="json")

    assert isinstance(dumped["id"], str)
    assert isinstance(dumped["started_at"], str)
    assert dumped["status"] == "running"
    assert dumped["finished_at"] is None


def test_python_dump_keeps_native_types():
    run = AgentRun(task="t")
    dumped = run.model_dump()

    assert isinstance(dumped["started_at"], datetime)
    # a plain python dump still round-trips back into an equal model
    assert AgentRun.model_validate(dumped) == run

"""Tests for :class:`agentlens.models.AgentRun`."""

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentlens.models import AgentRun, RunStatus

_STARTED = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _run(**overrides):
    base = {"task": "t"}
    base.update(overrides)
    return AgentRun(**base)


# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------


def test_valid_creation_with_defaults():
    run = AgentRun(task="summarise the document")

    assert isinstance(run.id, UUID)
    assert run.task == "summarise the document"
    assert run.status is RunStatus.RUNNING
    assert run.started_at.tzinfo is not None
    assert run.finished_at is None
    assert run.metadata == {}


def test_id_accepts_uuid_compatible_string():
    raw = str(uuid4())
    assert AgentRun(id=raw, task="t").id == UUID(raw)


def test_invalid_id_is_rejected():
    with pytest.raises(ValidationError):
        AgentRun(id="not-a-uuid", task="t")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        AgentRun(task="t", totally_new_field=1)


# ---------------------------------------------------------------------------
# Lifecycle: status <-> finished_at invariants
# ---------------------------------------------------------------------------


def test_running_without_finished_at_is_valid():
    run = _run(status=RunStatus.RUNNING)
    assert run.status is RunStatus.RUNNING
    assert run.finished_at is None


def test_running_defaults_are_valid():
    # default status is RUNNING and default finished_at is None
    assert _run(started_at=_STARTED).finished_at is None


def test_running_with_finished_at_is_rejected():
    with pytest.raises(ValidationError, match="finished_at must be None while status is RUNNING"):
        _run(
            status=RunStatus.RUNNING,
            started_at=_STARTED,
            finished_at=_STARTED + timedelta(seconds=1),
        )


def test_success_with_finished_at_is_valid():
    run = _run(
        status=RunStatus.SUCCESS,
        started_at=_STARTED,
        finished_at=_STARTED + timedelta(seconds=5),
    )
    assert run.status is RunStatus.SUCCESS
    assert run.finished_at > run.started_at


def test_success_without_finished_at_is_rejected():
    with pytest.raises(ValidationError, match="finished_at is required when status is SUCCESS"):
        _run(status=RunStatus.SUCCESS, started_at=_STARTED)


def test_failed_with_finished_at_is_valid():
    run = _run(
        status=RunStatus.FAILED,
        started_at=_STARTED,
        finished_at=_STARTED,
        metadata={"error": "ToolTimeout"},
    )
    assert run.status is RunStatus.FAILED
    # equal timestamps are allowed (a run may finish within timer resolution)
    assert run.finished_at == run.started_at


def test_failed_without_finished_at_is_rejected():
    with pytest.raises(ValidationError, match="finished_at is required when status is FAILED"):
        _run(status=RunStatus.FAILED, started_at=_STARTED)


@pytest.mark.parametrize("status", [RunStatus.SUCCESS, RunStatus.FAILED])
def test_completed_run_with_finished_before_started_is_rejected(status):
    with pytest.raises(ValidationError, match="finished_at must not be earlier than started_at"):
        _run(status=status, started_at=_STARTED, finished_at=_STARTED - timedelta(seconds=1))


# ---------------------------------------------------------------------------
# Timezone handling (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_naive_started_at_is_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        _run(started_at=datetime(2026, 1, 1, 12, 0))


def test_naive_finished_at_is_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        _run(
            status=RunStatus.SUCCESS,
            started_at=_STARTED,
            finished_at=datetime(2026, 1, 1, 12, 1),
        )


def test_non_utc_timestamps_are_normalised_to_utc():
    eastern = timezone(timedelta(hours=-5))
    started = datetime(2026, 1, 1, 7, 0, tzinfo=eastern)  # 12:00 UTC
    run = _run(started_at=started)

    assert run.started_at.utcoffset() == timedelta(0)
    assert run.started_at == _STARTED


def test_non_utc_finished_at_is_normalised_and_compared_in_utc():
    eastern = timezone(timedelta(hours=-5))
    run = _run(
        status=RunStatus.SUCCESS,
        started_at=_STARTED,
        finished_at=datetime(2026, 1, 1, 7, 0, 30, tzinfo=eastern),  # 12:00:30 UTC
    )
    assert run.finished_at == _STARTED + timedelta(seconds=30)


def test_iso_string_timestamps_are_accepted():
    run = _run(started_at="2026-01-01T12:00:00+00:00")
    assert run.started_at == _STARTED


# ---------------------------------------------------------------------------
# validate_assignment keeps the invariants after construction
# ---------------------------------------------------------------------------


def test_assignment_running_to_success_requires_finished_at():
    run = _run(status=RunStatus.RUNNING, started_at=_STARTED)
    with pytest.raises(ValidationError, match="finished_at is required when status is SUCCESS"):
        run.status = RunStatus.SUCCESS


def test_completing_a_run_via_full_revalidation_is_allowed():
    # Because both fields must change together, completion is expressed as a
    # revalidated copy rather than two separate attribute assignments.
    run = _run(status=RunStatus.RUNNING, started_at=_STARTED)
    completed = AgentRun.model_validate(
        run.model_dump()
        | {"status": RunStatus.SUCCESS, "finished_at": _STARTED + timedelta(seconds=1)}
    )
    assert completed.status is RunStatus.SUCCESS
    assert completed.finished_at == _STARTED + timedelta(seconds=1)
    assert completed.id == run.id


def test_assignment_setting_finished_at_while_running_is_rejected():
    run = _run(status=RunStatus.RUNNING, started_at=_STARTED)
    with pytest.raises(ValidationError, match="finished_at must be None while status is RUNNING"):
        run.finished_at = _STARTED + timedelta(seconds=1)

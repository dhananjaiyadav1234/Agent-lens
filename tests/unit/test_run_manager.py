"""Direct tests for RunManager lifecycle rules and atomic transitions."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agentlens.core import InMemoryTraceStore, RunLifecycleError, RunManager
from agentlens.models import RunStatus


def _manager():
    store = InMemoryTraceStore()
    return RunManager(store), store


def test_start_run_is_running_with_no_finish():
    mgr, _ = _manager()
    run = mgr.start_run("t")
    assert run.status is RunStatus.RUNNING
    assert run.finished_at is None
    assert run.started_at.tzinfo is not None


def test_complete_run_sets_success_and_finished_at_together():
    mgr, store = _manager()
    run = mgr.start_run("t")
    completed = mgr.complete_run(run.id)

    assert completed.status is RunStatus.SUCCESS
    assert completed.finished_at is not None
    assert completed.id == run.id
    assert completed.started_at == run.started_at
    # stored run is replaced, not mutated in place
    assert store.get_run(run.id) is completed
    assert store.get_run(run.id) is not run


def test_fail_run_sets_failed_and_finished_at_together():
    mgr, _ = _manager()
    run = mgr.start_run("t")
    failed = mgr.fail_run(run.id)
    assert failed.status is RunStatus.FAILED
    assert failed.finished_at is not None


def test_explicit_finished_at_is_respected():
    mgr, _ = _manager()
    run = mgr.start_run("t")
    ts = run.started_at + timedelta(seconds=3)
    completed = mgr.complete_run(run.id, finished_at=ts)
    assert completed.finished_at == ts


@pytest.mark.parametrize("first", ["complete", "fail"])
@pytest.mark.parametrize("second", ["complete", "fail"])
def test_a_terminal_run_cannot_transition_again(first, second):
    mgr, _ = _manager()
    run = mgr.start_run("t")
    getattr(mgr, f"{first}_run")(run.id)

    with pytest.raises(RunLifecycleError, match="already"):
        getattr(mgr, f"{second}_run")(run.id)


def test_transition_unknown_run_raises():
    mgr, _ = _manager()
    with pytest.raises(RunLifecycleError, match="unknown run"):
        mgr.complete_run(uuid4())


def test_transitioned_run_still_satisfies_model_invariants():
    mgr, _ = _manager()
    run = mgr.start_run("t")
    completed = mgr.complete_run(run.id)

    # re-validating the produced model must not raise (no reliance on a
    # temporarily invalid intermediate state)
    revalidated = completed.model_validate(completed.model_dump())
    assert revalidated == completed


def test_backdated_finished_at_before_started_at_is_rejected_by_model():
    mgr, _ = _manager()
    run = mgr.start_run("t")
    with pytest.raises(ValidationError, match="not be earlier"):
        mgr.complete_run(run.id, finished_at=datetime(2000, 1, 1, tzinfo=UTC))

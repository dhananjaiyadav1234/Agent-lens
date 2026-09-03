"""Tests for :class:`agentlens.models.AgentIssue`."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentlens.models import AgentIssue, IssueType, Severity


def _make(**overrides):
    base = {
        "run_id": uuid4(),
        "issue_type": IssueType.AGENT_LOOP,
        "severity": Severity.HIGH,
        "description": "The agent repeated the same tool call 6 times.",
    }
    base.update(overrides)
    return AgentIssue(**base)


def test_valid_creation():
    run_id = uuid4()
    issue = AgentIssue(
        run_id=run_id,
        issue_type=IssueType.TOOL_FAILURE,
        severity=Severity.CRITICAL,
        description="Tool 'db.query' failed 3 times in a row.",
    )
    assert isinstance(issue.id, UUID)
    assert issue.run_id == run_id
    assert issue.issue_type is IssueType.TOOL_FAILURE
    assert issue.severity is Severity.CRITICAL


def test_related_event_ids_defaults_to_empty_list():
    a = _make()
    b = _make()
    assert a.related_event_ids == []
    b.related_event_ids.append(uuid4())
    # separate instances must not share a mutable default
    assert a.related_event_ids == []


def test_related_event_ids_accepts_uuid_compatible_strings():
    raw = [str(uuid4()), str(uuid4())]
    issue = _make(related_event_ids=raw)
    assert issue.related_event_ids == [UUID(x) for x in raw]


def test_related_event_ids_rejects_invalid_uuid():
    with pytest.raises(ValidationError):
        _make(related_event_ids=["not-a-uuid"])


def test_invalid_run_id_is_rejected():
    with pytest.raises(ValidationError):
        _make(run_id="nope")


def test_invalid_issue_type_is_rejected():
    with pytest.raises(ValidationError):
        _make(issue_type="mystery")


def test_invalid_severity_is_rejected():
    with pytest.raises(ValidationError):
        _make(severity="apocalyptic")


def test_metadata_handling():
    issue = _make(metadata={"detector": "loop-v1", "score": 0.92, "window": [3, 9]})
    assert issue.metadata["detector"] == "loop-v1"

    with pytest.raises(ValidationError):
        _make(metadata={"bad": object()})


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        _make(extra=1)

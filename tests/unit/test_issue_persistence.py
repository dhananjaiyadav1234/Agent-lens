"""Issue-persistence unit tests for both trace stores."""

from uuid import uuid4

import pytest

from agentlens.core.storage import InMemoryTraceStore, TraceStore
from agentlens.models import AgentIssue, IssueType, Severity
from agentlens.storage import SQLiteTraceStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "issues.db"


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryTraceStore()
    else:
        s = SQLiteTraceStore(tmp_path / "store.db")
        try:
            yield s
        finally:
            s.close()


def _issue(run_id, *, issue_type=IssueType.AGENT_LOOP, severity=Severity.MEDIUM, **overrides):
    base = {
        "run_id": run_id,
        "issue_type": issue_type,
        "severity": severity,
        "description": "something happened",
    }
    base.update(overrides)
    return AgentIssue(**base)


# ---------------------------------------------------------------------------
# A. Protocol conformance
# ---------------------------------------------------------------------------


def test_both_stores_conform_to_trace_store_after_adding_issue_methods(db_path):
    assert isinstance(InMemoryTraceStore(), TraceStore)
    sqlite_store = SQLiteTraceStore(db_path)
    try:
        assert isinstance(sqlite_store, TraceStore)
    finally:
        sqlite_store.close()


# ---------------------------------------------------------------------------
# B / C. Shared store behaviour (parametrized over both implementations)
# ---------------------------------------------------------------------------


def test_empty_store_returns_empty_list(store):
    assert store.get_issues(uuid4()) == []
    assert store.list_issues() == []


def test_save_and_get_one_issue(store):
    run_id = uuid4()
    issue = _issue(run_id)
    store.save_issues([issue])
    assert store.get_issues(run_id) == [issue]


def test_save_and_get_multiple_issues_preserves_order(store):
    run_id = uuid4()
    issues = [
        _issue(run_id, description="first"),
        _issue(run_id, description="second"),
        _issue(run_id, description="third"),
    ]
    store.save_issues(issues)
    assert [i.description for i in store.get_issues(run_id)] == ["first", "second", "third"]


def test_issues_are_isolated_by_run_id(store):
    a, b = uuid4(), uuid4()
    store.save_issues([_issue(a, description="a-issue")])
    store.save_issues([_issue(b, description="b-issue")])
    assert [i.description for i in store.get_issues(a)] == ["a-issue"]
    assert [i.description for i in store.get_issues(b)] == ["b-issue"]


def test_unknown_run_returns_empty_list(store):
    store.save_issues([_issue(uuid4())])
    assert store.get_issues(uuid4()) == []


def test_get_issues_returns_a_fresh_list(store):
    run_id = uuid4()
    store.save_issues([_issue(run_id)])
    first = store.get_issues(run_id)
    first.append("mutation")
    assert len(store.get_issues(run_id)) == 1  # internal list untouched


def test_empty_save_issues_is_a_no_op(store):
    run_id = uuid4()
    store.save_issues([])
    assert store.get_issues(run_id) == []
    assert store.list_issues() == []


def test_list_issues_is_first_save_order(store):
    r1, r2 = uuid4(), uuid4()
    store.save_issues([_issue(r1, description="1")])
    store.save_issues([_issue(r2, description="2"), _issue(r1, description="3")])
    store.save_issues([_issue(r2, description="4")])
    assert [i.description for i in store.list_issues()] == ["1", "2", "3", "4"]


def test_saving_the_same_issue_object_twice_appends_twice(store):
    run_id = uuid4()
    issue = _issue(run_id)
    store.save_issues([issue])
    store.save_issues([issue])
    stored = store.get_issues(run_id)
    assert len(stored) == 2
    assert stored[0] == stored[1] == issue


def test_stored_issues_are_not_mutated(store):
    run_id = uuid4()
    issue = _issue(run_id, related_event_ids=[uuid4(), uuid4()], metadata={"k": [1, 2]})
    before = issue.model_dump()
    store.save_issues([issue])
    _ = store.get_issues(run_id)
    assert issue.model_dump() == before


# ---------------------------------------------------------------------------
# C. SQLite-specific: serialization round trips
# ---------------------------------------------------------------------------


def test_sqlite_schema_is_created_automatically(db_path):
    SQLiteTraceStore(db_path).close()
    assert db_path.exists()
    SQLiteTraceStore(db_path).close()  # re-open must not fail on CREATE


@pytest.mark.parametrize(
    "issue_type",
    [
        IssueType.AGENT_LOOP,
        IssueType.EXCESSIVE_RETRY,
        IssueType.DUPLICATE_TOOL_CALL,
        IssueType.INEFFICIENCY,
    ],
)
@pytest.mark.parametrize("severity", list(Severity))
def test_sqlite_issue_enums_round_trip(tmp_path, issue_type, severity):
    store = SQLiteTraceStore(tmp_path / "s.db")
    try:
        run_id = uuid4()
        issue = _issue(run_id, issue_type=issue_type, severity=severity)
        store.save_issues([issue])
        (restored,) = store.get_issues(run_id)
        assert restored == issue
        assert restored.issue_type is issue_type
        assert restored.severity is severity
    finally:
        store.close()


def test_sqlite_issue_full_round_trip_with_nested_metadata_and_related_ids(tmp_path):
    store = SQLiteTraceStore(tmp_path / "s.db")
    try:
        run_id = uuid4()
        related = [uuid4(), uuid4(), uuid4()]
        issue = AgentIssue(
            id=uuid4(),
            run_id=run_id,
            issue_type=IssueType.INEFFICIENCY,
            severity=Severity.CRITICAL,
            description="deeply structured",
            related_event_ids=list(related),
            metadata={
                "detector": "x",
                "evidence": ["a", "b"],
                "nested": {"n": [1, 2.5, None, True], "flag": False},
                "count": 0,
            },
        )
        store.save_issues([issue])
        (restored,) = store.get_issues(run_id)
        assert restored == issue
        assert restored.model_dump() == issue.model_dump()
        assert restored.related_event_ids == related  # exact order
        assert restored.metadata == issue.metadata
    finally:
        store.close()


def test_sqlite_issues_survive_close_and_reopen(db_path):
    run_id = uuid4()
    store_a = SQLiteTraceStore(db_path)
    saved = [_issue(run_id, description="one"), _issue(run_id, description="two")]
    store_a.save_issues(saved)
    store_a.close()
    del store_a

    store_b = SQLiteTraceStore(db_path)
    try:
        restored = store_b.get_issues(run_id)
        assert restored == saved
        assert [i.description for i in restored] == ["one", "two"]
        assert store_b.list_issues() == saved
    finally:
        store_b.close()


def test_sqlite_close_is_idempotent(db_path):
    store = SQLiteTraceStore(db_path)
    store.close()
    store.close()


def test_sqlite_separate_database_files_do_not_leak_issues(tmp_path):
    a = SQLiteTraceStore(tmp_path / "a.db")
    b = SQLiteTraceStore(tmp_path / "b.db")
    run_id = uuid4()
    a.save_issues([_issue(run_id)])
    try:
        assert len(a.get_issues(run_id)) == 1
        assert b.get_issues(run_id) == []
        assert b.list_issues() == []
    finally:
        a.close()
        b.close()


def test_sqlite_context_manager_closes(db_path):
    with SQLiteTraceStore(db_path) as store:
        store.save_issues([_issue(uuid4())])
    with pytest.raises(Exception):  # noqa: B017 - sqlite3.ProgrammingError on closed connection
        store.list_issues()

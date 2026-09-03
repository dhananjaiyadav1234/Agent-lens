"""Event ordering and sequence-number guarantees."""

from agentlens import AgentLens
from agentlens.models import EventType


def test_sequence_numbers_start_at_zero():
    lens = AgentLens()
    with lens.trace("t") as trace:
        pass
    assert lens.get_events(trace.run_id)[0].sequence_number == 0


def test_sequence_numbers_increase_by_one_and_are_unique():
    lens = AgentLens()
    with lens.trace("t") as trace:
        for i in range(10):
            trace.record_event(event_type=EventType.DECISION, name=f"e{i}")

    seqs = [e.sequence_number for e in lens.get_events(trace.run_id)]
    assert seqs == list(range(12))  # RUN_STARTED + 10 + RUN_COMPLETED
    assert len(seqs) == len(set(seqs))


def test_run_started_always_sequence_zero_even_with_many_events():
    lens = AgentLens()
    with lens.trace("t") as trace:
        trace.record_event(event_type=EventType.ERROR, name="handled")
    events = lens.get_events(trace.run_id)
    assert events[0].event_type is EventType.RUN_STARTED
    assert events[0].sequence_number == 0


def test_separate_runs_have_independent_sequences():
    lens = AgentLens()
    with lens.trace("first") as t1:
        t1.record_event(event_type=EventType.DECISION, name="a")
        t1.record_event(event_type=EventType.DECISION, name="b")

    with lens.trace("second") as t2:
        t2.record_event(event_type=EventType.DECISION, name="c")

    assert [e.sequence_number for e in lens.get_events(t1.run_id)] == [0, 1, 2, 3]
    assert [e.sequence_number for e in lens.get_events(t2.run_id)] == [0, 1, 2]


def test_all_event_types_are_accepted():
    lens = AgentLens()
    with lens.trace("t") as trace:
        for et in EventType:
            trace.record_event(event_type=et, name=et.value)
    recorded = {e.event_type for e in lens.get_events(trace.run_id)}
    assert set(EventType).issubset(recorded)


def test_timestamps_need_not_be_monotonic_but_sequence_is_authoritative():
    lens = AgentLens()
    with lens.trace("t") as trace:
        for i in range(5):
            trace.record_event(event_type=EventType.DECISION, name=f"e{i}")

    events = lens.get_events(trace.run_id)
    # ordering is by sequence number regardless of wall-clock resolution
    assert events == sorted(events, key=lambda e: e.sequence_number)
    assert [e.sequence_number for e in events] == list(range(7))

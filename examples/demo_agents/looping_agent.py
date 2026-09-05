"""Scenario B -- a deterministic agent loop.

The agent repeats the same two-step reasoning cycle (``analyze_request`` ->
``search_for_information``) three times with identical inputs, makes no progress,
then deliberately abandons the loop and finishes successfully.

Per cycle the event *names* and *inputs* are identical; only ``metadata.iteration``
differs, so a future loop detector can key on (name, normalized input) repetition
while still seeing how many times the cycle ran.
"""

from __future__ import annotations

from uuid import UUID

from agentlens import AgentLens
from agentlens.models import EventType

_LOOP_CYCLES = 3
_REQUEST = "Where is my order?"
_ANALYZE_INPUT = {"request": _REQUEST}
_SEARCH_INPUT = {"query": "order status", "request": _REQUEST}


def run(lens: AgentLens) -> UUID:
    """Execute the looping scenario on ``lens`` and return the new run id."""

    with lens.trace("Investigate a vague order-status request") as trace:
        for iteration in range(_LOOP_CYCLES):
            trace.record_event(
                event_type=EventType.DECISION,
                name="analyze_request",
                input=dict(_ANALYZE_INPUT),
                output={"conclusion": "need more information", "next": "search_for_information"},
                metadata={"iteration": iteration},
            )
            trace.record_event(
                event_type=EventType.DECISION,
                name="search_for_information",
                input=dict(_SEARCH_INPUT),
                output={"results": [], "next": "analyze_request"},
                metadata={"iteration": iteration},
            )

        trace.record_event(
            event_type=EventType.DECISION,
            name="abandon_search_and_respond",
            input={"attempts": _LOOP_CYCLES},
            output={
                "response": "I could not determine the order status from available information.",
                "reason": "repeated analyze/search cycle produced no new information",
            },
            metadata={"loop_detected_by_agent": False},
        )

    return trace.run_id

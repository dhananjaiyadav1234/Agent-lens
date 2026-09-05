"""Scenario A -- a healthy, efficient agent run.

Flow::

    RUN_STARTED
      -> DECISION  choose_lookup_strategy
      -> TOOL_CALL_STARTED    lookup_customer
      -> TOOL_CALL_COMPLETED  lookup_customer
      -> DECISION  prepare_response
    RUN_COMPLETED

No repeated decision cycle, no retries, no duplicate tool call, no wasted work.
This is the "no issues expected" baseline for future detectors.
"""

from __future__ import annotations

from uuid import UUID

from agentlens import AgentLens
from agentlens.models import EventType
from examples.demo_agents.tools import lookup_customer

_CUSTOMER_ID = "123"


def run(lens: AgentLens) -> UUID:
    """Execute the healthy scenario on ``lens`` and return the new run id."""

    with lens.trace("Answer a question about one customer's open orders") as trace:
        trace.record_event(
            event_type=EventType.DECISION,
            name="choose_lookup_strategy",
            input={"customer_id": _CUSTOMER_ID},
            output={"strategy": "direct_id_lookup"},
            metadata={"reason": "a concrete customer id is available"},
        )

        trace.record_event(
            event_type=EventType.TOOL_CALL_STARTED,
            name="lookup_customer",
            input={"customer_id": _CUSTOMER_ID},
        )
        result = lookup_customer(_CUSTOMER_ID)
        trace.record_event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name="lookup_customer",
            input={"customer_id": _CUSTOMER_ID},
            output=result,
            status="ok",
        )

        trace.record_event(
            event_type=EventType.DECISION,
            name="prepare_response",
            input={"customer_id": _CUSTOMER_ID, "open_orders": result["open_orders"]},
            output={"response": "Customer 123 has 2 open orders."},
            metadata={"information_sufficient": True},
        )

    return trace.run_id

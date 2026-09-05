"""Scenario E -- unnecessary work / poor execution strategy.

The agent already has everything it needs after one targeted ``lookup_customer``
call, but then broadens its scope, calls ``fetch_all_customers``, summarizes the
whole customer base, finally realizes only one customer was required, and
responds. The run finishes with ``SUCCESS``.

This is not a loop, a retry, or a duplicate tool call: it is a single pass that
does strictly more work than the task requires. Every wasteful step carries
metadata describing its ``purpose`` and the ``required_scope`` vs ``actual_scope``
gap so a future detector or AI analyzer can explain the inefficiency.
"""

from __future__ import annotations

from uuid import UUID

from agentlens import AgentLens
from agentlens.models import EventType
from examples.demo_agents.tools import fetch_all_customers, lookup_customer

_CUSTOMER_ID = "123"


def run(lens: AgentLens) -> UUID:
    """Execute the inefficient scenario on ``lens`` and return the new run id."""

    with lens.trace("Answer a single-customer question the long way") as trace:
        trace.record_event(
            event_type=EventType.DECISION,
            name="determine_information_needed",
            input={"goal": "report open orders for customer 123"},
            output={"needs": ["customer_123_profile"]},
            metadata={"required_scope": "single_customer"},
        )

        trace.record_event(
            event_type=EventType.TOOL_CALL_STARTED,
            name="lookup_customer",
            input={"customer_id": _CUSTOMER_ID},
        )
        target = lookup_customer(_CUSTOMER_ID)
        trace.record_event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name="lookup_customer",
            input={"customer_id": _CUSTOMER_ID},
            output=target,
            status="ok",
            metadata={"sufficient_to_answer": True},
        )

        trace.record_event(
            event_type=EventType.DECISION,
            name="reconsider_information_scope",
            input={"have": "customer_123_profile"},
            output={"decision": "gather the full customer base for context"},
            metadata={
                "purpose": "broaden context before answering",
                "required_scope": "single_customer",
                "chosen_scope": "all_customers",
                "necessary": False,
            },
        )

        trace.record_event(
            event_type=EventType.TOOL_CALL_STARTED,
            name="fetch_all_customers",
            input={},
            metadata={
                "purpose": "retrieve all customers",
                "required_scope": "single_customer",
                "actual_scope": "all_customers",
            },
        )
        everyone = fetch_all_customers()
        trace.record_event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name="fetch_all_customers",
            input={},
            output=everyone,
            status="ok",
            metadata={"rows_returned": everyone["count"], "rows_needed": 1},
        )

        trace.record_event(
            event_type=EventType.DECISION,
            name="summarize_all_customers",
            input={"customer_count": everyone["count"]},
            output={"summary": f"Reviewed {everyone['count']} customer records."},
            metadata={
                "purpose": "summarize retrieved data",
                "useful_fraction": "1 of 3",
                "wasted_work": True,
            },
        )

        trace.record_event(
            event_type=EventType.DECISION,
            name="recognize_over_retrieval",
            input={"reviewed": everyone["count"], "required": 1},
            output={"realization": "only customer 123 was needed"},
            metadata={
                "wasted_operation": "fetch_all_customers",
                "wasted_scope": "all_customers",
                "required_scope": "single_customer",
            },
        )

        trace.record_event(
            event_type=EventType.DECISION,
            name="prepare_response",
            input={"customer_id": _CUSTOMER_ID, "open_orders": target["open_orders"]},
            output={"response": "Customer 123 has 2 open orders."},
            metadata={"answer_derived_from": "lookup_customer"},
        )

    return trace.run_id

"""Scenario D -- an unnecessary duplicate tool call.

The agent calls ``lookup_customer`` with ``{"customer_id": "123"}``, gets a
complete answer, then -- without any new information -- calls the exact same tool
with the exact same input again before responding. The run finishes with
``SUCCESS``.

The two calls share tool name and normalized input; the second call's metadata
records that it depends on no new information, which is what a future
duplicate-work detector would confirm from ordering + input equality.
"""

from __future__ import annotations

from uuid import UUID

from agentlens import AgentLens
from agentlens.models import EventType
from examples.demo_agents.tools import lookup_customer

_CUSTOMER_ID = "123"
_TOOL = "lookup_customer"
_TOOL_INPUT = {"customer_id": _CUSTOMER_ID}


def _call_lookup(trace, *, redundant: bool) -> dict[str, object]:
    trace.record_event(
        event_type=EventType.TOOL_CALL_STARTED,
        name=_TOOL,
        input=dict(_TOOL_INPUT),
        metadata={"redundant": redundant, "depends_on_new_information": False} if redundant else {},
    )
    result = lookup_customer(_CUSTOMER_ID)
    trace.record_event(
        event_type=EventType.TOOL_CALL_COMPLETED,
        name=_TOOL,
        input=dict(_TOOL_INPUT),
        output=result,
        status="ok",
        metadata={"redundant": redundant} if redundant else {},
    )
    return result


def run(lens: AgentLens) -> UUID:
    """Execute the duplicate-tool scenario on ``lens`` and return the new run id."""

    with lens.trace("Look up a customer, then look them up again") as trace:
        trace.record_event(
            event_type=EventType.DECISION,
            name="determine_customer_lookup",
            input={"customer_id": _CUSTOMER_ID},
            output={"tool": _TOOL, "tool_input": dict(_TOOL_INPUT)},
        )
        _call_lookup(trace, redundant=False)

        trace.record_event(
            event_type=EventType.DECISION,
            name="double_check_customer_details",
            input={"customer_id": _CUSTOMER_ID},
            output={"decision": "re-run the same lookup", "uses_new_information": False},
            metadata={"justified": False},
        )
        result = _call_lookup(trace, redundant=True)

        trace.record_event(
            event_type=EventType.DECISION,
            name="prepare_response",
            input={"customer_id": _CUSTOMER_ID, "open_orders": result["open_orders"]},
            output={"response": "Customer 123 has 2 open orders."},
        )

    return trace.run_id

"""Scenario C -- an excessive retry pattern.

The agent tries the same logical operation (``fetch_customer`` for customer 123)
four times. The first three attempts "fail" and are recorded as structured
``ERROR`` events; the fourth succeeds. The run finishes with ``SUCCESS``.

The simulated failures are recorded as ordinary trace events -- the ``with
lens.trace(...)`` block never sees an exception, so the run is *successful but
inefficient*, not failed.
"""

from __future__ import annotations

from uuid import UUID

from agentlens import AgentLens
from agentlens.models import EventType
from examples.demo_agents.tools import lookup_customer

_CUSTOMER_ID = "123"
_FAILED_ATTEMPTS = 3
_OPERATION = "fetch_customer"
_ATTEMPT_INPUT = {"customer_id": _CUSTOMER_ID}


def run(lens: AgentLens) -> UUID:
    """Execute the retry scenario on ``lens`` and return the new run id."""

    with lens.trace("Fetch a customer record through a flaky tool") as trace:
        trace.record_event(
            event_type=EventType.DECISION,
            name="plan_customer_fetch",
            input={"customer_id": _CUSTOMER_ID},
            output={"operation": _OPERATION, "retry_budget": _FAILED_ATTEMPTS + 1},
        )

        for attempt in range(1, _FAILED_ATTEMPTS + 1):
            trace.record_event(
                event_type=EventType.TOOL_CALL_STARTED,
                name=_OPERATION,
                input=dict(_ATTEMPT_INPUT),
                metadata={"attempt": attempt},
            )
            trace.record_event(
                event_type=EventType.ERROR,
                name="temporary_tool_failure",
                input=dict(_ATTEMPT_INPUT),
                output={
                    "operation": _OPERATION,
                    "error_type": "TemporaryError",
                    "message": "temporary failure",
                    "attempt": attempt,
                },
                status="error",
                metadata={"attempt": attempt, "recoverable": True},
            )

        final_attempt = _FAILED_ATTEMPTS + 1
        trace.record_event(
            event_type=EventType.TOOL_CALL_STARTED,
            name=_OPERATION,
            input=dict(_ATTEMPT_INPUT),
            metadata={"attempt": final_attempt},
        )
        result = lookup_customer(_CUSTOMER_ID)
        trace.record_event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name=_OPERATION,
            input=dict(_ATTEMPT_INPUT),
            output=result,
            status="ok",
            metadata={"attempt": final_attempt},
        )

        trace.record_event(
            event_type=EventType.DECISION,
            name="prepare_response",
            input={"customer_id": _CUSTOMER_ID, "open_orders": result["open_orders"]},
            output={"response": "Customer 123 has 2 open orders."},
            metadata={"total_attempts": final_attempt, "failed_attempts": _FAILED_ATTEMPTS},
        )

    return trace.run_id

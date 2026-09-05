"""Capture LangChain activity into an AgentLens run, then detect and report.

Fully offline: it invokes a local ``langchain_core`` tool (no LLM, no network,
no API key) with the AgentLens callback handler attached, then uses the existing
detection / reporting pipeline. Nothing is written to the repository.

Usage::

    python -m examples.langchain_integration
"""

from __future__ import annotations

import contextlib
from uuid import uuid4

from langchain_core.outputs import Generation, LLMResult
from langchain_core.tools import tool

from agentlens import AgentLens
from agentlens.integrations.langchain import AgentLensCallbackHandler

_ATTEMPTS = {"n": 0}


@tool
def fetch_customer(customer_id: str) -> dict:
    """Fetch a customer record -- flaky on the first two calls (offline, deterministic)."""

    _ATTEMPTS["n"] += 1
    if _ATTEMPTS["n"] <= 3:
        raise RuntimeError("temporary upstream failure")
    return {"id": customer_id, "open_orders": 2}


def main() -> None:
    lens = AgentLens()

    with lens.trace("Answer: where is my order?") as trace:
        handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)

        # a (fake, local) planning LLM call
        llm_run = uuid4()
        handler.on_llm_start(
            {"name": "planner"}, ["Which tool answers order status?"], run_id=llm_run
        )
        handler.on_llm_end(
            LLMResult(generations=[[Generation(text="call fetch_customer")]]), run_id=llm_run
        )

        # the tool fails three times, then succeeds
        for _ in range(3):
            with contextlib.suppress(RuntimeError):
                fetch_customer.invoke({"customer_id": "123"}, config={"callbacks": [handler]})
        fetch_customer.invoke({"customer_id": "123"}, config={"callbacks": [handler]})

    run_id = trace.run_id

    events = lens.get_events(run_id)
    print("recorded events:")
    for event in events:
        print(f"  {event.sequence_number}: {event.event_type.value} / {event.name}")

    issues = lens.detect(run_id)  # detection is explicit
    report = lens.get_report(run_id)  # reporting is explicit

    print(f"\nrun status: {report.run.status.value}")
    print(f"total events: {report.summary.total_events}")
    print(f"total issues: {report.summary.total_issues}")
    for issue in issues:
        print(
            f"  {issue.issue_type.value} / {issue.severity.value} / "
            f"operation={issue.metadata.get('operation')} "
            f"failed_attempts={issue.metadata.get('failed_attempts')}"
        )


if __name__ == "__main__":
    main()

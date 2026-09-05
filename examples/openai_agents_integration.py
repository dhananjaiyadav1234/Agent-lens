"""Record OpenAI Agents SDK span activity into an AgentLens run, then detect and report.

Fully offline: it drives the SDK's real tracing machinery (``agents.tracing.trace``
+ span factories) with no model, no API key, and no network. It writes to a
throwaway SQLite database that is removed on exit.

Usage::

    python -m examples.openai_agents_integration
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from agents.tracing import function_span, generation_span, set_trace_processors
from agents.tracing import trace as agents_trace

from agentlens import AgentLens
from agentlens.integrations.openai_agents import AgentLensOpenAITracer
from agentlens.storage import SQLiteTraceStore


def _failing_tool(name: str, args: dict) -> None:
    with function_span(name=name, input=json.dumps(args)) as span:
        span.set_error(
            {"message": "temporary upstream failure", "data": {"exception_type": "RuntimeError"}}
        )


def _ok_tool(name: str, args: dict, output: dict) -> None:
    with function_span(name=name, input=json.dumps(args)) as span:
        span.span_data.output = output


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlens-openai-"))
    db_path = workdir / "agentlens.db"
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)

        with lens.trace("Answer: where is my order?") as trace:
            tracer = AgentLensOpenAITracer(lens=lens, run_id=trace.run_id)
            set_trace_processors([tracer])

            with agents_trace("customer-support"):
                with generation_span(model="planner-model") as span:
                    span.span_data.output = [
                        {"role": "assistant", "content": "call fetch_customer"}
                    ]
                for _ in range(3):
                    _failing_tool("fetch_customer", {"customer_id": "123"})
                _ok_tool("fetch_customer", {"customer_id": "123"}, {"open_orders": 2})

            set_trace_processors([])

        run_id = trace.run_id

        for event in lens.get_events(run_id):
            print(f"  {event.sequence_number}: {event.event_type.value} / {event.name}")

        issues = lens.detect(run_id)  # explicit
        report = lens.get_report(run_id)  # explicit

        print(f"\nrun status: {report.run.status.value}")
        print(f"total events: {report.summary.total_events}")
        print(f"total issues: {report.summary.total_issues}")
        for issue in issues:
            print(
                f"  {issue.issue_type.value} / {issue.severity.value} / "
                f"operation={issue.metadata.get('operation')} "
                f"failed_attempts={issue.metadata.get('failed_attempts')}"
            )
    finally:
        set_trace_processors([])
        store.close()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

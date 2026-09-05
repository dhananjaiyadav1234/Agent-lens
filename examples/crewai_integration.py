"""Record CrewAI event-bus activity into an AgentLens run, then detect and report.

Fully offline: it drives CrewAI's real event bus (``crewai_event_bus.emit`` with
the SDK's own event classes) with no crew, no model, no API key, and no network.
It writes to a throwaway SQLite database that is removed on exit.

Usage::

    python -m examples.crewai_integration
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from pathlib import Path

import crewai.events.event_listener as _crewai_event_listener
from crewai.events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

from agentlens import AgentLens
from agentlens.integrations.crewai import AgentLensCrewAIListener
from agentlens.storage import SQLiteTraceStore

# Keep CrewAI's built-in console listener quiet for a readable demo.
_crewai_event_listener.event_listener.formatter.verbose = False


def _llm_call(model: str, prompt: str, answer: str) -> None:
    crewai_event_bus.emit(
        "demo",
        LLMCallStartedEvent(
            call_id="c1", model=model, messages=[{"role": "user", "content": prompt}]
        ),
    )
    crewai_event_bus.emit(
        "demo",
        LLMCallCompletedEvent(call_id="c1", model=model, response=answer, call_type="llm_call"),
    )


def _tool_failure(name: str, args: dict) -> None:
    crewai_event_bus.emit("demo", ToolUsageStartedEvent(tool_name=name, tool_args=args))
    crewai_event_bus.emit(
        "demo",
        ToolUsageErrorEvent(tool_name=name, tool_args=args, error="temporary upstream failure"),
    )


def _tool_success(name: str, args: dict, output: str) -> None:
    now = dt.datetime.now(tz=dt.UTC)
    crewai_event_bus.emit("demo", ToolUsageStartedEvent(tool_name=name, tool_args=args))
    crewai_event_bus.emit(
        "demo",
        ToolUsageFinishedEvent(
            tool_name=name, tool_args=args, output=output, started_at=now, finished_at=now
        ),
    )


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlens-crewai-"))
    db_path = workdir / "agentlens.db"
    store = SQLiteTraceStore(db_path)
    try:
        lens = AgentLens(store=store)

        with (
            lens.trace("Research: where is my order?") as trace,
            AgentLensCrewAIListener(lens=lens, run_id=trace.run_id),
        ):
            _llm_call("planner-model", "plan the work", "call fetch_customer")
            for _ in range(3):
                _tool_failure("fetch_customer", {"customer_id": "123"})
            _tool_success("fetch_customer", {"customer_id": "123"}, '{"open_orders": 2}')

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
        store.close()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

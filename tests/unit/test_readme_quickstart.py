"""Executes the README Quick Start workflow verbatim.

Keeps the documented example honest: if core APIs used here change shape, this
test breaks before the README silently goes stale.
"""

from __future__ import annotations

import json

from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.models import AgentReport, EventType


def test_readme_quickstart_workflow():
    lens = AgentLens()

    with lens.trace("Look up a customer's order status") as trace:
        trace.record_event(
            event_type=EventType.TOOL_CALL_STARTED,
            name="lookup_customer",
            input={"customer_id": "123"},
        )
        trace.record_event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            name="lookup_customer",
            input={"customer_id": "123"},
            output={"open_orders": 2},
            status="ok",
        )

    run_id = trace.run_id

    issues = lens.detect(run_id)
    report = lens.get_report(run_id)

    assert report.summary.total_events == 4
    assert issues == []

    json_report = export_json(report)
    markdown_report = export_markdown(report)

    assert json.loads(json_report) == report.model_dump(mode="json")
    assert AgentReport.model_validate(json.loads(json_report)) == report
    assert "# AgentLens Report" in markdown_report
    assert "lookup_customer" in markdown_report

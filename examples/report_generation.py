"""Trace a scenario, detect issues, then build a read-only aggregate report.

Fully offline and deterministic. Uses the default in-memory store -- no files
are written.

Usage::

    python -m examples.report_generation
"""

from __future__ import annotations

from agentlens import AgentLens
from examples.demo_agents import run_inefficient_agent


def main() -> None:
    lens = AgentLens()
    run_id = run_inefficient_agent(lens)
    lens.detect(run_id)

    # get_report is read-only: issues are unchanged by it.
    issues_before = lens.get_issues(run_id)
    report = lens.get_report(run_id)
    issues_after = lens.get_issues(run_id)
    assert issues_before == issues_after, "get_report must not modify storage"

    summary = report.summary
    print(f"Run:              {report.run.id}")
    print(f"Task:             {report.run.task}")
    print(f"Status:           {report.run.status.value}")
    print(f"Event count:      {summary.total_events}")
    print(f"Issue count:      {summary.total_issues}")
    print("Events by type:")
    for event_type, count in summary.events_by_type.items():
        print(f"    {event_type.value}: {count}")
    print("Issues by type:")
    for issue_type, count in summary.issues_by_type.items():
        print(f"    {issue_type.value}: {count}")
    print("Issues by severity:")
    for severity, count in summary.issues_by_severity.items():
        print(f"    {severity.value}: {count}")

    print("\nget_report() is read-only: issue count before == after:", len(issues_before))


if __name__ == "__main__":
    main()

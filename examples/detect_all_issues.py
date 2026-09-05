"""Run all deterministic detectors over every demo scenario via ``lens.detect``.

Offline and deterministic: no network, no API keys, no external services.

Usage::

    python -m examples.detect_all_issues
"""

from __future__ import annotations

from agentlens import AgentLens
from examples.demo_agents import SCENARIOS


def main() -> None:
    lens = AgentLens()

    for name, scenario in SCENARIOS.items():
        run_id = scenario(lens)
        issues = lens.detect(run_id)

        print(f"{name}: {len(issues)} issue(s)")
        for issue in issues:
            print(
                f"  {issue.issue_type.value} / {issue.severity.value} / "
                f"{issue.description} "
                f"(related_events={len(issue.related_event_ids)})"
            )


if __name__ == "__main__":
    main()

"""Run every demo scenario and report what the DuplicateToolDetector finds.

Offline and deterministic: no network, no API keys, no external services.

Usage::

    python -m examples.detect_duplicate_tools
"""

from __future__ import annotations

from agentlens import AgentLens
from agentlens.detectors import DuplicateToolDetector
from examples.demo_agents import SCENARIOS


def main() -> None:
    lens = AgentLens()
    detector = DuplicateToolDetector()

    for name, scenario in SCENARIOS.items():
        run_id = scenario(lens)
        run = lens.get_run(run_id)
        events = lens.get_events(run_id)
        issues = detector.detect(run, events)

        print(f"{name}: {len(issues)} duplicate tool issue(s)")
        for issue in issues:
            print(
                f"  {issue.issue_type.value} / {issue.severity.value} / "
                f"operation={issue.metadata['operation']} "
                f"related_events={len(issue.related_event_ids)}"
            )


if __name__ == "__main__":
    main()

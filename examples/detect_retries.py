"""Run every demo scenario and report what the RetryDetector finds.

Offline and deterministic: no network, no API keys, no external services.

Usage::

    python -m examples.detect_retries
"""

from __future__ import annotations

from agentlens import AgentLens
from agentlens.detectors import RetryDetector
from examples.demo_agents import SCENARIOS


def main() -> None:
    lens = AgentLens()
    detector = RetryDetector()

    for name, scenario in SCENARIOS.items():
        run_id = scenario(lens)
        run = lens.get_run(run_id)
        events = lens.get_events(run_id)
        issues = detector.detect(run, events)

        print(f"{name}: {len(issues)} retry issue(s)")
        for issue in issues:
            print(
                f"  {issue.issue_type.value} / {issue.severity.value} / "
                f"operation={issue.metadata['operation']} "
                f"failed_attempts={issue.metadata['failed_attempts']} "
                f"related_events={len(issue.related_event_ids)}"
            )


if __name__ == "__main__":
    main()

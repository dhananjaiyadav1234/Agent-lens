"""Run every demo agent scenario against one AgentLens instance and print a summary.

Fully offline and deterministic: no network, no API keys, no paid services.

Usage::

    python -m examples.run_demo_scenarios
"""

from __future__ import annotations

from collections import Counter

from agentlens import AgentLens
from examples.demo_agents import SCENARIOS


def main() -> None:
    lens = AgentLens()

    print(f"{'scenario':<13} {'status':<8} {'events':>6}  event types")
    print("-" * 72)

    for name, scenario in SCENARIOS.items():
        run_id = scenario(lens)
        run = lens.get_run(run_id)
        events = lens.get_events(run_id)
        type_counts = Counter(event.event_type.value for event in events)
        summary = ", ".join(f"{etype}x{count}" for etype, count in sorted(type_counts.items()))
        print(f"{name:<13} {run.status.value:<8} {len(events):>6}  {summary}")

    print("-" * 72)
    print(f"{len(SCENARIOS)} scenarios, {len(lens.list_completed_runs())} completed runs")


if __name__ == "__main__":
    main()

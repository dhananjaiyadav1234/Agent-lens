"""Controlled demo agent scenarios for AgentLens.

Each scenario is a deterministic, offline function ``run(lens: AgentLens) -> UUID``
that drives the real public tracing API (``with lens.trace(...) as trace: ...``)
to produce one completed run, and returns its ``run_id``. No scenario creates its
own ``AgentLens`` -- the caller supplies and owns the instance.

Scenarios
---------

============== ================================================================
``normal``     Healthy, efficient run -- the "no issues expected" baseline.
``looping``    Repeats an analyze/search decision cycle 3x, then finishes.
``retrying``   Retries one tool operation, 3 structured ERROR events then success.
``duplicate``  Calls the same tool with the same input twice, unnecessarily.
``inefficient`` Over-retrieves (fetches all customers when one was needed).
============== ================================================================

All problematic scenarios still finish with ``status == SUCCESS``: they model
*successful but suboptimal* execution, not failed runs.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from agentlens import AgentLens
from examples.demo_agents.duplicate_tool_agent import run as run_duplicate_tool_agent
from examples.demo_agents.inefficient_agent import run as run_inefficient_agent
from examples.demo_agents.looping_agent import run as run_looping_agent
from examples.demo_agents.normal_agent import run as run_normal_agent
from examples.demo_agents.retrying_agent import run as run_retrying_agent

Scenario = Callable[[AgentLens], UUID]

SCENARIOS: dict[str, Scenario] = {
    "normal": run_normal_agent,
    "looping": run_looping_agent,
    "retrying": run_retrying_agent,
    "duplicate": run_duplicate_tool_agent,
    "inefficient": run_inefficient_agent,
}

__all__ = [
    "Scenario",
    "SCENARIOS",
    "run_normal_agent",
    "run_looping_agent",
    "run_retrying_agent",
    "run_duplicate_tool_agent",
    "run_inefficient_agent",
]

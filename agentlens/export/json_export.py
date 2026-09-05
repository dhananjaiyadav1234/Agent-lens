"""Deterministic JSON export of an :class:`~agentlens.models.AgentReport`."""

from __future__ import annotations

import json

from agentlens.models import AgentReport

__all__ = ["export_json"]


def export_json(report: AgentReport) -> str:
    """Serialize ``report`` to a deterministic, pretty-printed JSON string.

    The payload is exactly ``report.model_dump(mode="json")`` -- UUIDs become
    strings, datetimes ISO 8601 strings, enums their values -- rendered with
    ``sort_keys=True``, ``indent=2`` and ``ensure_ascii=False``. Because keys are
    sorted, the output is byte-identical across repeated calls regardless of dict
    construction order; ``json.loads`` of the result equals
    ``report.model_dump(mode="json")``.

    Pure: ``report`` is not mutated and no validation is bypassed.
    """

    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )

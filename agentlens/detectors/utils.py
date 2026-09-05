"""Deterministic helpers shared by detectors.

Nothing here does I/O, mutates its arguments, or depends on anything outside the
universal models and the standard library.
"""

from __future__ import annotations

import json

from agentlens.models import AgentEvent
from agentlens.models.base import JsonValue

__all__ = ["canonical_json", "event_fingerprint", "operation_fingerprint"]


def canonical_json(value: JsonValue | None) -> str:
    """Serialize a JSON-compatible value to a canonical string.

    Two values that are semantically equal produce byte-identical output:

    * dictionary keys are sorted, recursively, so insertion order is irrelevant;
    * list order is preserved (it is semantically meaningful);
    * separators are fixed so whitespace never varies;
    * JSON primitive distinctions are kept (``1``, ``1.0``, ``"1"``, ``true``,
      ``null`` all serialize differently).

    The input is never mutated. The exact string form is an internal detail and
    not part of any public API.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def event_fingerprint(event: AgentEvent) -> str:
    """Return a stable logical identity for an event.

    The fingerprint is built from exactly three fields:

    * ``event_type``
    * ``name``
    * the canonical JSON of ``input``

    Everything else -- ``id``, ``run_id``, ``timestamp``, ``sequence_number``,
    ``duration_ms``, ``status``, ``output`` and ``metadata`` -- is deliberately
    excluded, because those fields legitimately vary between repetitions of the
    same logical operation (the looping demo, for example, varies
    ``metadata.iteration`` across otherwise-identical cycles).
    """

    return canonical_json([event.event_type.value, event.name, event.input])


def operation_fingerprint(name: str, tool_input: JsonValue | None) -> str:
    """Return a stable logical identity for a *tool operation*.

    The identity is exactly the tool ``name`` plus the canonical JSON of its
    ``input`` -- nothing else. It deliberately ignores ``event_type`` so a
    ``TOOL_CALL_STARTED`` and the ``TOOL_CALL_COMPLETED`` that resolves it share
    an identity, and ignores ids, timestamps, sequence numbers, metadata,
    duration and output, which vary between attempts of the same operation.

    Dictionary key order does not affect the result; list order and JSON
    primitive distinctions do. The input is never mutated.
    """

    return canonical_json([name, tool_input])

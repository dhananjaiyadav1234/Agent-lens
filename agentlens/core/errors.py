"""Exceptions raised by the AgentLens core.

The hierarchy is intentionally shallow: a single base plus two concrete errors
that separate *where you are in the trace lifecycle* problems from *how a run may
change state* problems.
"""

from __future__ import annotations

__all__ = ["AgentLensError", "TraceStateError", "RunLifecycleError"]


class AgentLensError(Exception):
    """Base class for every error raised by AgentLens."""


class TraceStateError(AgentLensError):
    """The tracing context is in the wrong state for the requested operation.

    Raised when recording an event without an active trace, recording on a trace
    that has already closed, or opening a nested trace.
    """


class RunLifecycleError(AgentLensError):
    """An invalid run lifecycle transition was requested.

    Raised when completing or failing a run that is already in a terminal state
    (``SUCCESS`` or ``FAILED``), or transitioning a run that does not exist.
    """

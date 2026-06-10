"""Current invocation context accessor."""

from __future__ import annotations

from dispatch_agents._internal import dispatch as _dispatch
from dispatch_agents.models import InvocationContext as _InvocationContext

__all__ = ["current"]


def current() -> _InvocationContext | None:
    """Return the current invocation context.

    Returns:
        :class:`dispatch_agents.models.InvocationContext` with the current
        ``trace_id``, ``invocation_id``, and optional ``parent_id``, or ``None``
        when called outside a Dispatch invocation.

    Example::

        from dispatch_agents.context import current

        ctx = current()
        if ctx is not None:
            print(ctx.trace_id, ctx.invocation_id)
    """
    trace_id = _dispatch.get_current_trace_id()
    invocation_id = _dispatch.get_current_invocation_id()
    if trace_id is None or invocation_id is None:
        return None
    return _InvocationContext(
        trace_id=trace_id,
        invocation_id=invocation_id,
        parent_id=_dispatch.get_current_parent_id(),
    )

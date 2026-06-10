from __future__ import annotations

import logging
import traceback
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ValidationError

import dispatch_agents.handlers as _handlers
from dispatch_agents._internal.models import (
    BaseMessage,
    ErrorPayload,
    FunctionMessage,
    Message,
    SuccessPayload,
    TopicMessage,
)

_logger = logging.getLogger(__name__)

_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
_current_invocation_id: ContextVar[str | None] = ContextVar(
    "current_invocation_id", default=None
)
_current_parent_id: ContextVar[str | None] = ContextVar(
    "current_parent_id", default=None
)
_current_message: ContextVar[BaseMessage | None] = ContextVar(
    "current_message", default=None
)

_trace_invocation_context: OrderedDict[str, str] = OrderedDict()
_TRACE_CONTEXT_MAX_SIZE = 100000


def _set_current_message(message: BaseMessage) -> None:
    """Set current contextvars for the message being dispatched."""
    _current_trace_id.set(message.trace_id)
    _current_invocation_id.set(message.uid)
    _current_parent_id.set(message.parent_id)
    _current_message.set(message)


def get_current_trace_id() -> str | None:
    """Get the current trace ID from execution context."""
    return _current_trace_id.get()


def get_current_invocation_id() -> str | None:
    """Get the current invocation ID from execution context."""
    return _current_invocation_id.get()


def get_current_parent_id() -> str | None:
    """Get the current parent ID from execution context."""
    return _current_parent_id.get()


def get_invocation_id_for_trace(trace_id: str | None) -> str | None:
    """Look up invocation ID by trace ID."""
    if trace_id is None:
        return None
    return _trace_invocation_context.get(trace_id)


def _register_trace_invocation(trace_id: str, invocation_id: str) -> None:
    """Register a trace_id -> invocation_id mapping with bounded size."""
    if trace_id in _trace_invocation_context:
        _trace_invocation_context.move_to_end(trace_id)
        _trace_invocation_context[trace_id] = invocation_id
        return

    _trace_invocation_context[trace_id] = invocation_id
    while len(_trace_invocation_context) > _TRACE_CONTEXT_MAX_SIZE:
        _trace_invocation_context.popitem(last=False)


def _unregister_trace_invocation(trace_id: str) -> None:
    """Remove a trace_id -> invocation_id mapping."""
    _trace_invocation_context.pop(trace_id, None)


async def _dispatch_message(message: Message) -> SuccessPayload | ErrorPayload:
    """
    Called by the agent's gRPC server when a message is received.
    Routes to the appropriate handler based on message type:
    - TopicMessage: looks up handler via handlers._TOPIC_HANDLERS[topic]
    - FunctionMessage: looks up handler directly via handlers._REGISTERED_HANDLERS[function_name]

    All handlers registered through @on or @fn are callable via FunctionMessage.
    TopicMessage routes event-delivered payloads to @on handlers.

    Returns:
        SuccessPayload: When handler executes successfully, contains the return value
        ErrorPayload: When handler raises an exception, contains error details
    """
    # Set context for the duration of this message processing
    _set_current_message(message)

    # Register trace -> invocation mapping for fallback lookup.
    # This helps when external SDKs (OpenAI, Claude) don't properly propagate
    # Python context variables to their tool call contexts.
    if message.trace_id and message.uid:
        _register_trace_invocation(message.trace_id, message.uid)

    try:
        # Route based on message type
        if isinstance(message, TopicMessage):
            if (
                message.topic not in _handlers._TOPIC_HANDLERS
                or not _handlers._TOPIC_HANDLERS[message.topic]
            ):
                raise ValueError(f"No handler registered for topic: {message.topic}")
            handler_names = _handlers._TOPIC_HANDLERS[message.topic]
        elif isinstance(message, FunctionMessage):
            if message.function_name not in _handlers._REGISTERED_HANDLERS:
                raise ValueError(f"No handler registered: {message.function_name}")
            handler_names = [message.function_name]
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")

        # Call all handlers for the topic (fan-out pattern)
        # Returns the last successful result, or the last error if all fail
        last_result: SuccessPayload | ErrorPayload | None = None

        for handler_name in handler_names:
            func = _handlers._REGISTERED_HANDLERS[handler_name]
            # Extract input model from handler function's type hints
            input_model = _handlers._get_input_model_from_handler(func)

            try:
                # Validate payload against input schema
                if not input_model:
                    raise ValueError(
                        f"No input model found for handler: {handler_name}"
                    )

                payload_obj = input_model.model_validate(message.payload)

                # Call handler with validated payload
                raw_fn_return = await func(payload_obj)

                # Serialize return value
                if isinstance(raw_fn_return, BaseModel):
                    result = raw_fn_return.model_dump()
                elif raw_fn_return is None:
                    result = None
                else:
                    result = raw_fn_return

                last_result = SuccessPayload(result=result)

            except ValidationError as e:
                _logger.error(
                    "Validation error in event handler",
                    extra={
                        "handler": handler_name,
                        "error_type": "ValidationError",
                        "validation_errors": e.errors(),
                    },
                    exc_info=True,
                )
                # e.errors() can contain non-serializable objects (e.g. ValueError
                # instances in ctx). Convert to JSON-safe format via str().
                safe_details: list[dict[str, Any]] = []
                _json_safe = str | int | float | bool | list | dict | None
                for err in e.errors():
                    safe_err = {
                        k: str(v) if not isinstance(v, _json_safe) else v
                        for k, v in err.items()
                    }
                    safe_details.append(safe_err)
                last_result = ErrorPayload(
                    error=str(e),
                    error_type="ValidationError",
                    trace=traceback.format_exc(),
                    details=safe_details,
                )

            except Exception as e:
                _logger.error(
                    "Error in event handler",
                    extra={"handler": handler_name, "error": str(e)},
                    exc_info=True,
                )
                last_result = ErrorPayload(
                    error=str(e),
                    error_type=type(e).__name__,
                    trace=traceback.format_exc(),
                )

        # Should always have a result since we check handler_names is not empty
        assert last_result is not None
        return last_result

    finally:
        # Clean up trace context mapping when invocation completes.
        # This ensures deterministic cleanup rather than relying solely on LRU eviction.
        if message.trace_id:
            _unregister_trace_invocation(message.trace_id)


async def _run_init_hook() -> None:
    """Run the registered init hook if present.

    Called by the gRPC server before starting to handle requests.

    Raises:
        Exception: If the init hook fails, the exception is propagated.
    """
    if _handlers._INIT_HOOK is not None:
        _logger.info(f"Running @init function: {_handlers._INIT_HOOK.__name__}")
        await _handlers._INIT_HOOK()
        _logger.info(f"Completed @init function: {_handlers._INIT_HOOK.__name__}")

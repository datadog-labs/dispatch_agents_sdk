"""Agent handler registration decorators.

Use ``@fn`` for functions that other agents call directly with
:func:`dispatch_agents.invocation.invoke`, ``@on`` for event handlers, and
``@init`` for one-time async startup. Handler payloads must be Pydantic models;
the SDK extracts input and output schemas from type hints for validation and
capability registration.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, ParamSpec, TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from dispatch_agents.models import BasePayload as _BasePayload
from dispatch_agents.models import HandlerMetadata as _HandlerMetadata

__all__ = ["fn", "get_handler_metadata", "get_handler_schemas", "init", "on"]

# Preserve the decorated handler's exact signature so concretely-typed payloads
# and responses survive type checking (e.g. ``(WeatherRequest) -> WeatherResponse``).
_P = ParamSpec("_P")
_R = TypeVar("_R")

_AsyncHandler = Callable[[BaseModel], Awaitable[BaseModel | None]]
_REGISTERED_HANDLERS: dict[str, _AsyncHandler] = {}
_HANDLER_METADATA: dict[str, _HandlerMetadata] = {}
_TOPIC_HANDLERS: dict[str, list[str]] = {}
_INIT_HOOK: Callable[[], Awaitable[None]] | None = None


def _extract_return_model(return_type: Any) -> type[BaseModel] | None:
    """Extract BaseModel from return type, handling Optional/Union."""
    if not return_type:
        return None

    origin = get_origin(return_type)
    if origin is not None:
        args = get_args(return_type)
        for arg in args:
            if (
                arg is not type(None)
                and isinstance(arg, type)
                and issubclass(arg, BaseModel)
            ):
                return arg

    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type

    return None


def _get_input_model_from_handler(
    func: Callable[..., Any],
) -> type[BaseModel] | None:
    """Extract the input model type from a handler function's type hints."""
    try:
        hints = get_type_hints(func)
    except Exception:
        return None

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        return None

    first_param_type = hints.get(params[0].name)
    if first_param_type and isinstance(first_param_type, type):
        if issubclass(first_param_type, BaseModel):
            return first_param_type

    return None


def fn(
    *, name: str | None = None
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Register a function as directly callable by other agents.

    Functions registered with ``@fn`` can be called from another agent with
    :func:`dispatch_agents.invocation.invoke`. The first parameter must be
    annotated with a Pydantic model, typically a :class:`BasePayload` subclass.
    The return annotation, if present, defines the response schema.

    Args:
        name: Optional invocation name. Defaults to ``function.__name__``.

    Returns:
        A decorator that registers the callable while preserving its type hints.

    Raises:
        ValueError: If the invocation name is already registered.
        ValueError: If the first handler parameter is not annotated with a
            Pydantic model.

    Example::

        from dispatch_agents import BasePayload, fn

        class WeatherRequest(BasePayload):
            city: str

        class WeatherResponse(BasePayload):
            temperature: int

        @fn()
        async def get_weather(payload: WeatherRequest) -> WeatherResponse:
            return WeatherResponse(temperature=72)
    """

    def decorator(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        fn_name = name or func.__name__

        if fn_name in _REGISTERED_HANDLERS:
            raise ValueError(f"Handler already registered: {fn_name}")

        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        input_model: type[BaseModel] | None = None
        if params:
            first_param_type = hints.get(params[0].name)
            if first_param_type:
                if isinstance(first_param_type, type) and issubclass(
                    first_param_type, BaseModel
                ):
                    input_model = first_param_type

        if not input_model:
            raise ValueError(
                f"Handler '{fn_name}' must have a first parameter "
                f"annotated with a Pydantic BaseModel subclass. "
                f"Example: async def {fn_name}(payload: MyPayload) -> Result: ..."
            )

        return_type = hints.get("return")
        output_model = _extract_return_model(return_type)
        metadata = _HandlerMetadata(
            handler_name=fn_name,
            topics=[],
            input_schema=input_model.model_json_schema(mode="serialization"),
            output_schema=output_model.model_json_schema(mode="serialization")
            if output_model
            else None,
            handler_doc=func.__doc__,
        )

        func._dispatch_metadata = metadata  # type: ignore[attr-defined]
        _HANDLER_METADATA[fn_name] = metadata
        _REGISTERED_HANDLERS[fn_name] = func  # type: ignore[assignment]
        return func

    return decorator


def init(
    func: Callable[[], Awaitable[None]],
) -> Callable[[], Awaitable[None]]:
    """Register the agent's async initialization hook.

    The hook runs once in the agent event loop before the agent handles any
    request. Use it for async setup such as connecting MCP servers, initializing
    SDK clients, or loading shared state. Only one ``@init`` function can be
    registered per agent.

    Args:
        func: An async function with no parameters.

    Returns:
        The original function.

    Raises:
        TypeError: If ``func`` is not an async function.
        ValueError: If an init function is already registered.

    Example::

        from dispatch_agents import init
        from dispatch_agents.contrib.openai import get_mcp_servers
        from agents import Agent

        agent: Agent

        @init
        async def setup() -> None:
            global agent
            agent = Agent(
                name="assistant",
                mcp_servers=await get_mcp_servers(),
            )
    """
    if not asyncio.iscoroutinefunction(func):
        raise TypeError(f"@init function must be async: {func.__name__}")

    global _INIT_HOOK
    if _INIT_HOOK is not None:
        raise ValueError(
            f"Only one @init function allowed. Already registered: {_INIT_HOOK.__name__}"
        )
    _INIT_HOOK = func
    return func


def get_handler_schemas() -> dict[str, _HandlerMetadata]:
    """Return metadata for all registered handlers.

    Returns a mapping of handler names to :class:`dispatch_agents.models.HandlerMetadata`
    values, including input schema, output schema, subscribed topics, and handler
    docstring.

    This is primarily useful for capability inspection, tests, and tooling that
    needs to inspect the handlers registered in the current process.
    """
    return dict(_HANDLER_METADATA)


def get_handler_metadata(topic: str) -> _HandlerMetadata | None:
    """Return metadata for the first handler registered for ``topic``.

    Args:
        topic: Event topic to inspect.

    Returns:
        Handler metadata for the first matching topic handler, or ``None`` when
        no handler is registered for the topic.
    """
    handler_names = _TOPIC_HANDLERS.get(topic)
    if not handler_names:
        return None
    return _HANDLER_METADATA.get(handler_names[0])


def _validate_github_payload_compatibility(
    input_model: type[BaseModel],
    event_classes: list[type[BaseModel]],
    handler_name: str,
) -> None:
    """Validate handler payload type compatibility with GitHub events."""
    for event_cls in event_classes:
        if not issubclass(event_cls, input_model):
            raise TypeError(
                f"Handler '{handler_name}' payload type {input_model.__name__} "
                f"is not compatible with {event_cls.__name__}. "
                f"Use a common base class or the exact event class."
            )


def _dispatch_topic_for_event(event: type[BaseModel]) -> str:
    dispatch_topic = getattr(event, "dispatch_topic", None)
    if not callable(dispatch_topic):
        raise TypeError(
            f"Invalid github_event type: {event}. "
            "Expected a payload model class with a dispatch_topic() method."
        )
    topic = dispatch_topic()
    if not isinstance(topic, str):
        raise TypeError(
            f"Invalid github_event type: {event}. dispatch_topic() must return str."
        )
    return topic


def on(
    *,
    topic: str | None = None,
    github_event: type[_BasePayload] | Sequence[type[_BasePayload]] | None = None,
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Register an event handler for a topic or GitHub event payload type.

    The handler function must accept a Pydantic model payload. The decorator
    extracts input and output schemas from type hints for validation and API
    documentation. Handlers registered with ``@on`` are also directly callable by
    function name through :func:`dispatch_agents.invocation.invoke`.

    Multiple handlers can subscribe to the same topic. When an event arrives,
    the platform invokes all matching handlers concurrently; each handler runs in
    its own invocation and may succeed or fail independently.

    Args:
        topic: Custom event topic to handle, such as ``"user.created"``.
        github_event: GitHub event payload class, or sequence of classes, to
            subscribe to. Mutually exclusive with ``topic``.

    Returns:
        A decorator that registers the handler while preserving its type hints.

    Raises:
        ValueError: If both ``topic`` and ``github_event`` are specified.
        ValueError: If neither ``topic`` nor ``github_event`` is specified.
        ValueError: If the handler name is already registered.
        ValueError: If the first handler parameter is not annotated with a
            Pydantic model.
        TypeError: If ``github_event`` is not a GitHub event payload class or
            the handler payload type is incompatible with the selected events.

    Example::

        from dispatch_agents import BasePayload, on

        class UserCreated(BasePayload):
            user_id: str

        @on(topic="user.created")
        async def handle_user_created(payload: UserCreated) -> None:
            print(payload.user_id)

    Example::

        from dispatch_agents import on
        from dispatch_agents.integrations.github.events import PullRequestOpened

        @on(github_event=PullRequestOpened)
        async def handle_pr(payload: PullRequestOpened) -> None:
            print(payload.pull_request.title)
    """
    if topic and github_event:
        raise ValueError("Cannot specify both 'topic' and 'github_event'")
    if not topic and not github_event:
        raise ValueError("Must specify either 'topic' or 'github_event'")

    topics: list[str] = []
    github_event_classes: list[type[BaseModel]] = []

    if github_event:
        events = (
            [github_event] if isinstance(github_event, type) else list(github_event)
        )

        for event in events:
            if isinstance(event, type) and issubclass(event, BaseModel):
                github_event_classes.append(event)
                topics.append(_dispatch_topic_for_event(event))
            else:
                raise TypeError(
                    f"Invalid github_event type: {type(event)}. "
                    "Expected a payload model class with dispatch_topic()."
                )
    else:
        topics = [topic]  # type: ignore[list-item]

    def decorator(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        handler_name = func.__name__

        if handler_name in _REGISTERED_HANDLERS:
            existing_metadata = _HANDLER_METADATA[handler_name]
            for t in topics:
                if t not in existing_metadata.topics:
                    existing_metadata.topics.append(t)
                    if t not in _TOPIC_HANDLERS:
                        _TOPIC_HANDLERS[t] = []
                    if handler_name not in _TOPIC_HANDLERS[t]:
                        _TOPIC_HANDLERS[t].append(handler_name)
            return func

        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        input_model: type[BaseModel] | None = None
        if params:
            first_param_type = hints.get(params[0].name)
            if first_param_type:
                if isinstance(first_param_type, type) and issubclass(
                    first_param_type, BaseModel
                ):
                    input_model = first_param_type

        if not input_model:
            topic_desc = ", ".join(topics)
            raise ValueError(
                f"Handler for topic(s) '{topic_desc}' must have a first parameter "
                f"annotated with a Pydantic BaseModel subclass. "
                f"Example: async def handler(payload: MyPayload) -> Result: ..."
            )

        if github_event_classes:
            _validate_github_payload_compatibility(
                input_model, github_event_classes, handler_name
            )

        return_type = hints.get("return")
        output_model = _extract_return_model(return_type)
        metadata = _HandlerMetadata(
            handler_name=handler_name,
            topics=topics,
            input_schema=input_model.model_json_schema(mode="serialization"),
            output_schema=output_model.model_json_schema(mode="serialization")
            if output_model
            else None,
            handler_doc=func.__doc__,
        )

        func._dispatch_metadata = metadata  # type: ignore[attr-defined]
        _HANDLER_METADATA[handler_name] = metadata
        _REGISTERED_HANDLERS[handler_name] = func  # type: ignore[assignment]
        for t in topics:
            if t not in _TOPIC_HANDLERS:
                _TOPIC_HANDLERS[t] = []
            if handler_name not in _TOPIC_HANDLERS[t]:
                _TOPIC_HANDLERS[t].append(handler_name)

        return func

    return decorator

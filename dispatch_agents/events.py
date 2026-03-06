from __future__ import annotations

import asyncio
import inspect
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import httpx
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from dispatch_agents.integrations.github import GitHubEventPayload

from dispatch_agents.models import (
    BaseMessage,
    ErrorPayload,
    FunctionMessage,
    InvokeFunctionRequest,
    JsonSchema,
    Message,
    PublishEventBody,
    StrictBaseModel,
    SuccessPayload,
    TopicMessage,
)

from .version import get_sdk_version

logger = logging.getLogger(__name__)

# Type variable for generic payload types
PayloadT = TypeVar("PayloadT", bound=BaseModel)
ReturnT = TypeVar("ReturnT", bound=BaseModel | None)

# ParamSpec and TypeVar for preserving decorator function signatures
P = ParamSpec("P")
R = TypeVar("R")


# Base class for all event payloads - users must inherit from this
class BasePayload(StrictBaseModel):
    """Base class for all dispatch agent event payloads.

    All handler input parameters must inherit from this class to ensure
    proper type validation and schema extraction.

    Examples:
        >>> class MyEventPayload(BasePayload):
        ...     message: str
        ...     user_id: int
        ...
        >>> @on(topic="my.topic")
        ... async def my_handler(payload: MyEventPayload) -> str:
        ...     return f"Hello {payload.user_id}"
    """

    pass


class HandlerMetadata(StrictBaseModel):
    """Serializable handler metadata for registration and introspection.

    This model provides type-safe access to handler metadata including
    input/output schemas, topic subscriptions, and documentation.

    Note: input_model and output_model type references are not stored here
    since they can be extracted from the handler function via get_type_hints()
    when needed for validation.
    """

    handler_name: str
    topics: list[str]
    input_schema: JsonSchema
    output_schema: JsonSchema | None
    handler_doc: str | None


# Type alias for async handler functions - takes a BaseModel and returns BaseModel or None
AsyncHandler = Callable[[BaseModel], Awaitable[BaseModel | None]]

# Unified handler registry - maps handler_name -> handler function and metadata
# Used by both @on (topic-based) and @fn (direct call) decorators
# All handlers can be invoked directly by name; @on handlers additionally have topic triggers
REGISTERED_HANDLERS: dict[str, AsyncHandler] = {}
HANDLER_METADATA: dict[str, HandlerMetadata] = {}

# Topic-to-handler mapping for efficient topic routing
# Maps topic -> list of handler_names (used to look up handlers in REGISTERED_HANDLERS)
# Multiple handlers can subscribe to the same topic (fan-out pattern)
TOPIC_HANDLERS: dict[str, list[str]] = {}

# Init hook - async function called once when the agent starts
# Runs in the agent's event loop before handling any requests
_INIT_HOOK: Callable[[], Awaitable[None]] | None = None

# Thread-safe context variables for tracking current execution context
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

# Trace-indexed invocation context store.
# Maps trace_id -> invocation_id for fallback lookup when context variables
# aren't propagated (e.g., when external SDKs use separate async contexts).
# Uses OrderedDict to maintain insertion order for LRU-style eviction.
# Thread-safe for single operations in CPython due to GIL.
from collections import OrderedDict

_trace_invocation_context: OrderedDict[str, str] = OrderedDict()
_TRACE_CONTEXT_MAX_SIZE = 100000  # Maximum number of trace mappings to keep


def _register_trace_invocation(trace_id: str, invocation_id: str) -> None:
    """Register a trace_id -> invocation_id mapping with bounded size.

    Uses LRU-style eviction: when the cache is full, the oldest entries
    are removed to make room for new ones.

    Args:
        trace_id: The trace ID to register
        invocation_id: The invocation ID to associate with the trace
    """
    # If already exists, move to end (most recently used)
    if trace_id in _trace_invocation_context:
        _trace_invocation_context.move_to_end(trace_id)
        _trace_invocation_context[trace_id] = invocation_id
        return

    # Add new entry
    _trace_invocation_context[trace_id] = invocation_id

    # Evict oldest entries if over limit
    while len(_trace_invocation_context) > _TRACE_CONTEXT_MAX_SIZE:
        _trace_invocation_context.popitem(last=False)  # Remove oldest (first)


def _unregister_trace_invocation(trace_id: str) -> None:
    """Remove a trace_id -> invocation_id mapping.

    Called when an invocation completes to ensure deterministic cleanup.
    Silently ignores if the trace_id is not found.

    Args:
        trace_id: The trace ID to unregister
    """
    _trace_invocation_context.pop(trace_id, None)


def _extract_return_model(return_type: Any) -> type[BaseModel] | None:
    """Extract BaseModel from return type, handling Optional/Union."""
    if not return_type:
        return None

    # Check if it's Optional[Model] or Union[Model, None]
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

    # Direct BaseModel subclass
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type

    return None


def _get_input_model_from_handler(
    func: Callable[..., Any],
) -> type[BaseModel] | None:
    """Extract the input model type from a handler function's type hints.

    This is used at runtime to get the input model for payload validation,
    avoiding the need to store type references in serializable metadata.
    """
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
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Register a function as directly callable by other agents.

    Functions registered with @fn can be called directly using invoke().

    Args:
        name: Optional function name for invocation (defaults to function.__name__)

    Returns:
        A decorator function that registers the callable while preserving type hints

    Examples:
        >>> @fn()
        ... async def get_weather(payload: WeatherRequest) -> WeatherResponse:
        ...     return WeatherResponse(temp=72)
        ...
        >>> # Called from another agent:
        >>> result = await invoke("weather-agent", "get_weather", {"city": "NYC"})
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        fn_name = name or func.__name__

        if fn_name in REGISTERED_HANDLERS:
            raise ValueError(f"Handler already registered: {fn_name}")

        # Extract type information from function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Extract input model (first parameter that's a BaseModel subclass)
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

        # Extract output model
        return_type = hints.get("return")
        output_model = _extract_return_model(return_type)

        # Store unified metadata (type-safe Pydantic model)
        metadata = HandlerMetadata(
            handler_name=fn_name,
            topics=[],  # No topic subscriptions for @fn
            input_schema=input_model.model_json_schema(mode="serialization"),
            output_schema=output_model.model_json_schema(mode="serialization")
            if output_model
            else None,
            handler_doc=func.__doc__,
        )

        # Store metadata on function for introspection
        func._dispatch_metadata = metadata  # type: ignore

        # Register in unified registries
        HANDLER_METADATA[fn_name] = metadata
        REGISTERED_HANDLERS[fn_name] = func  # type: ignore[assignment]

        return func

    return decorator


def init(
    func: Callable[[], Awaitable[None]],
) -> Callable[[], Awaitable[None]]:
    """Register the agent's initialization function.

    The init function runs once in the agent's event loop before handling
    any requests. Use this for async initialization such as connecting to
    MCP servers or initializing database connections.

    Only one function can be decorated with @init per agent.

    Args:
        func: An async function with no parameters

    Returns:
        The original function (unmodified)

    Raises:
        ValueError: If an init function is already registered

    Examples:
        >>> from dispatch_agents.contrib.openai import get_mcp_servers
        >>> from agents import Agent
        >>>
        >>> my_agent: Agent  # Module-level, initialized by @init
        >>>
        >>> @init
        ... async def setup():
        ...     mcp_servers = await get_mcp_servers()
        ...     global my_agent
        ...     my_agent = Agent(name="MyAgent", mcp_servers=mcp_servers)
        >>>
        >>> @on(topic="query")
        ... async def handle_query(payload: QueryRequest) -> QueryResponse:
        ...     result = await Runner.run(my_agent, payload.prompt)
        ...     return QueryResponse(result=result.final_output)
    """
    global _INIT_HOOK

    if not asyncio.iscoroutinefunction(func):
        raise TypeError(f"@init function must be async: {func.__name__}")

    if _INIT_HOOK is not None:
        raise ValueError(
            f"Only one @init function allowed. Already registered: {_INIT_HOOK.__name__}"
        )

    _INIT_HOOK = func
    return func


def _validate_github_payload_compatibility(
    input_model: type[BaseModel],
    event_classes: list[type[GitHubEventPayload]],
    handler_name: str,
) -> None:
    """Validate handler's payload type is compatible with subscribed GitHub event classes.

    The handler's input model must be a base class (or exact match) of all event classes.
    This ensures type safety when handling multiple event types with a common base.

    Args:
        input_model: The handler's input payload model
        event_classes: List of GitHub event classes the handler subscribes to
        handler_name: Name of the handler (for error messages)

    Raises:
        TypeError: If the handler's payload type is not compatible with all events
    """
    for event_cls in event_classes:
        # The input model should be the same class or a base class of the event
        if not issubclass(event_cls, input_model):
            raise TypeError(
                f"Handler '{handler_name}' payload type {input_model.__name__} "
                f"is not compatible with {event_cls.__name__}. "
                f"Use a common base class or the exact event class."
            )


def on(
    *,
    topic: str | None = None,
    github_event: type[GitHubEventPayload]
    | list[type[GitHubEventPayload]]
    | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Register an event handler for a topic or GitHub event(s).

    The handler function should accept a payload parameter that is a Pydantic BaseModel
    subclass. The decorator will automatically extract input/output schemas from the
    function's type hints and register them for validation and API documentation.

    Handlers registered with @on can also be called directly using invoke() by their
    function name, just like @fn handlers.

    Args:
        topic: The event topic to handle (e.g., "user.created")
        github_event: GitHub event(s) to subscribe to. Mutually exclusive with topic.
            Can be a single event class (e.g., PullRequestOpened) or a list of classes.

    Returns:
        A decorator function that registers the handler while preserving type hints

    Examples:
        # Subscribe to a custom topic
        >>> @on(topic="user.created")
        ... async def handle_user_created(payload: UserCreatedPayload) -> WelcomeEmailPayload:
        ...     return WelcomeEmailPayload(...)

        # Subscribe to a GitHub event
        >>> from dispatch_agents.integrations.github import PullRequestOpened
        >>> @on(github_event=PullRequestOpened)
        ... async def handle_pr(payload: PullRequestOpened) -> None:
        ...     print(f"PR opened: {payload.pull_request.title}")

        # Subscribe to multiple GitHub events
        >>> from dispatch_agents.integrations.github import (
        ...     PullRequestOpened, PullRequestSynchronize, PullRequestBase
        ... )
        >>> @on(github_event=[PullRequestOpened, PullRequestSynchronize])
        ... async def handle_pr_changes(payload: PullRequestBase) -> None:
        ...     ...
    """
    # Deferred import required to avoid circular import:
    # events.py -> github/__init__.py -> events.py (for BasePayload)
    from dispatch_agents.integrations.github import GitHubEventPayload

    # Validate parameters
    if topic and github_event:
        raise ValueError("Cannot specify both 'topic' and 'github_event'")
    if not topic and not github_event:
        raise ValueError("Must specify either 'topic' or 'github_event'")

    # Convert github_event to list of topics
    topics: list[str] = []
    github_event_classes: list[type[GitHubEventPayload]] = []

    if github_event:
        events = github_event if isinstance(github_event, list) else [github_event]

        for event in events:
            if isinstance(event, type) and issubclass(event, GitHubEventPayload):
                github_event_classes.append(event)
                topics.append(event.dispatch_topic())
            else:
                raise TypeError(
                    f"Invalid github_event type: {type(event)}. "
                    f"Expected a GitHub event class (e.g., PullRequestOpened)."
                )
    else:
        topics = [topic]  # type: ignore[list-item]

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        handler_name = func.__name__

        if handler_name in REGISTERED_HANDLERS:
            # Handler exists - check if we're just adding more topics to it
            existing_metadata = HANDLER_METADATA[handler_name]
            for t in topics:
                if t not in existing_metadata.topics:
                    existing_metadata.topics.append(t)
                    if t not in TOPIC_HANDLERS:
                        TOPIC_HANDLERS[t] = []
                    if handler_name not in TOPIC_HANDLERS[t]:
                        TOPIC_HANDLERS[t].append(handler_name)
            return func

        # Extract type information from function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Extract input model (first parameter that's a BaseModel subclass)
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

        # Validate GitHub event payload compatibility
        if github_event_classes:
            _validate_github_payload_compatibility(
                input_model, github_event_classes, handler_name
            )

        # Extract output model (return type if it's a BaseModel subclass)
        return_type = hints.get("return")
        output_model = _extract_return_model(return_type)

        # Store unified metadata (type-safe Pydantic model)
        metadata = HandlerMetadata(
            handler_name=handler_name,
            topics=topics,
            input_schema=input_model.model_json_schema(mode="serialization"),
            output_schema=output_model.model_json_schema(mode="serialization")
            if output_model
            else None,
            handler_doc=func.__doc__,
        )

        # Store metadata on the function
        func._dispatch_metadata = metadata  # type: ignore

        # Register in unified registries
        HANDLER_METADATA[handler_name] = metadata
        REGISTERED_HANDLERS[handler_name] = func  # type: ignore[assignment]
        for t in topics:
            if t not in TOPIC_HANDLERS:
                TOPIC_HANDLERS[t] = []
            if handler_name not in TOPIC_HANDLERS[t]:
                TOPIC_HANDLERS[t].append(handler_name)

        return func

    return decorator


async def dispatch_message(message: Message) -> SuccessPayload | ErrorPayload:
    """
    Called by the agent's gRPC server when a message is received.
    Routes to the appropriate handler based on message type:
    - TopicMessage: looks up handler via TOPIC_HANDLERS[topic]
    - FunctionMessage: looks up handler directly via REGISTERED_HANDLERS[function_name]

    All handlers (from @on and @fn) are callable via FunctionMessage.
    TopicMessage routing is maintained for backwards compatibility with existing workflows.

    Returns:
        SuccessPayload: When handler executes successfully, contains the return value
        ErrorPayload: When handler raises an exception, contains error details
    """
    import traceback

    # Set context for the duration of this message processing
    _current_trace_id.set(message.trace_id)
    _current_invocation_id.set(message.uid)
    _current_parent_id.set(message.parent_id)
    _current_message.set(message)

    # Register trace -> invocation mapping for fallback lookup.
    # This helps when external SDKs (OpenAI, Claude) don't properly propagate
    # Python context variables to their tool call contexts.
    if message.trace_id and message.uid:
        _register_trace_invocation(message.trace_id, message.uid)

    try:
        # Route based on message type
        if isinstance(message, TopicMessage):
            if message.topic not in TOPIC_HANDLERS or not TOPIC_HANDLERS[message.topic]:
                raise ValueError(f"No handler registered for topic: {message.topic}")
            handler_names = TOPIC_HANDLERS[message.topic]
        elif isinstance(message, FunctionMessage):
            if message.function_name not in REGISTERED_HANDLERS:
                raise ValueError(f"No handler registered: {message.function_name}")
            handler_names = [message.function_name]
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")

        # Call all handlers for the topic (fan-out pattern)
        # Returns the last successful result, or the last error if all fail
        last_result: SuccessPayload | ErrorPayload | None = None

        for handler_name in handler_names:
            func = REGISTERED_HANDLERS[handler_name]
            # Extract input model from handler function's type hints
            input_model = _get_input_model_from_handler(func)

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
                logger.error(
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
                logger.error(
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


def get_current_trace_id() -> str | None:
    """Get the current trace ID from execution context."""
    return _current_trace_id.get()


def get_current_invocation_id() -> str | None:
    """Get the current invocation ID from execution context.

    The invocation ID uniquely identifies the current message/request being processed.
    This is the most specific identifier for correlating downstream calls (like MCP
    tool invocations) with the parent agent invocation.
    """
    return _current_invocation_id.get()


def get_invocation_id_for_trace(trace_id: str | None) -> str | None:
    """Look up invocation ID by trace ID.

    This is a fallback mechanism for when Python context variables aren't properly
    propagated (e.g., when external SDKs like OpenAI Agents or Claude Agent SDK
    execute tool calls in separate async contexts).

    Args:
        trace_id: The trace ID to look up

    Returns:
        The invocation ID associated with the trace, or None if not found.
    """
    if trace_id is None:
        return None
    return _trace_invocation_context.get(trace_id)


def get_current_parent_id() -> str | None:
    """Get the current parent ID from execution context.

    The parent ID identifies the message that triggered this invocation,
    useful for tracing chains of invocations.
    """
    return _current_parent_id.get()


async def run_init_hook() -> None:
    """Run the registered init hook if present.

    Called by the gRPC server before starting to handle requests.

    Raises:
        Exception: If the init hook fails, the exception is propagated.
    """
    if _INIT_HOOK is not None:
        logger.info(f"Running @init function: {_INIT_HOOK.__name__}")
        await _INIT_HOOK()
        logger.info(f"Completed @init function: {_INIT_HOOK.__name__}")


def get_handler_schemas() -> dict[str, HandlerMetadata]:
    """Get all registered handler schemas.

    Returns a dictionary mapping handler names to their metadata, including
    input/output schemas, topics (if any), and documentation.

    This is useful for:
    - Registering agent capabilities with the backend
    - Displaying available handlers in UI
    - Schema validation

    Returns:
        Dict mapping handler names to HandlerMetadata with fields:
        - handler_name: Name of the handler function
        - input_schema: JSON schema for input payload
        - output_schema: JSON schema for output payload (or None)
        - handler_doc: Docstring from the handler function
        - topics: List of topics this handler subscribes to (empty for @fn)
    """
    return dict(HANDLER_METADATA)


def get_handler_metadata(topic: str) -> HandlerMetadata | None:
    """Get metadata for a specific topic's handler.

    Args:
        topic: The topic to get metadata for

    Returns:
        HandlerMetadata for the handler, or None if topic not registered.
        If multiple handlers are registered for the topic, returns the first one's metadata.
    """
    handler_names = TOPIC_HANDLERS.get(topic)
    if not handler_names:
        return None
    # Return metadata for the first handler
    handler_name = handler_names[0]
    return HANDLER_METADATA.get(handler_name)


def _get_router_url() -> str:
    """Get the dispatch router URL from environment or default."""
    return os.getenv("BACKEND_URL", "http://dispatch.api:8000")


def _get_namespace() -> str | None:
    """Get the dispatch namespace from environment.

    Returns None if not set - the caller should handle the missing namespace.
    """
    return os.getenv("DISPATCH_NAMESPACE")


def _get_api_base_url() -> str:
    """Get the API base URL from environment or default.

    Raises RuntimeError if DISPATCH_NAMESPACE is not set.
    """
    namespace = _get_namespace()
    if not namespace:
        raise RuntimeError(
            "DISPATCH_NAMESPACE environment variable is required. "
            "Set it to the namespace your agent is deployed in."
        )
    return _get_router_url() + f"/api/unstable/namespace/{namespace}"


def _get_auth_headers() -> dict[str, str]:
    """Get authentication and version headers for API requests.

    Returns headers including Authorization and SDK version information.
    """

    headers = {
        "x-dispatch-client": "sdk",
        "x-dispatch-client-version": get_sdk_version(),
        "x-dispatch-client-commit": os.getenv("GIT_COMMIT", "unknown")[:8],
    }

    api_key = os.getenv("DISPATCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


async def emit_event(topic: str, payload: Any, sender_id: str | None = None) -> str:
    """Emit an event to the dispatch router.

    Args:
        topic: The topic/event type to publish to
        payload: The event payload data
        sender_id: Optional sender identifier (defaults to current agent)

    Returns:
        The unique event ID (uid) of the published message
    """
    if sender_id is None:
        sender_id = os.getenv("DISPATCH_AGENT_NAME", "unknown-agent")

    # Automatically inherit context from current execution
    trace_id = _current_trace_id.get()
    # Child events should point to the current handler (invocation_id) as their
    # parent, not the current handler's own parent.
    parent_id = _current_invocation_id.get()

    event_body = PublishEventBody(
        topic=topic,
        payload=payload if isinstance(payload, dict) else {"data": payload},
        sender_id=sender_id,
        trace_id=trace_id,
        parent_id=parent_id,
    )

    api_base_url = _get_api_base_url()
    auth_headers = _get_auth_headers()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_base_url}/events/publish",
            json=event_body.model_dump(),
            headers=auth_headers,
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("event_uid", str(uuid.uuid4()))


# Type variable for response_model generic
ResponseT = TypeVar("ResponseT", bound=BaseModel)


@overload
async def invoke(
    agent_name: str,
    function_name: str,
    payload: dict[str, Any] | BaseModel,
    *,
    response_model: type[ResponseT],
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> ResponseT: ...


@overload
async def invoke(
    agent_name: str,
    function_name: str,
    payload: dict[str, Any] | BaseModel,
    *,
    response_model: None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> dict[str, Any]: ...


async def invoke(
    agent_name: str,
    function_name: str,
    payload: dict[str, Any] | BaseModel,
    *,
    response_model: type[ResponseT] | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> ResponseT | dict[str, Any]:
    """Call a function on another agent and await the response.

    This enables direct function calls between agents in the same namespace.
    The target agent must have the function registered with @fn decorator.

    The function uses a polling pattern:
    1. POST to /invoke starts the invocation and returns an invocation_id
    2. GET /invoke/{invocation_id} polls until status is "completed" or "error"
    3. Returns the result when done

    For fire-and-forget calls, wrap in asyncio.create_task():
        asyncio.create_task(invoke("agent", "fn", payload))

    Args:
        agent_name: Name of the target agent
        function_name: Name of the function to call
        payload: Input data (dict or Pydantic BaseModel)
        response_model: Optional Pydantic model to validate and parse the response.
            When provided, returns an instance of the model instead of a dict.
        timeout: Maximum time to wait for completion in seconds (default 60)
        poll_interval: Time between status checks in seconds (default 0.5)

    Returns:
        The function's return value as a dict, or as an instance of response_model
        if provided.

    Raises:
        httpx.HTTPStatusError: If the backend returns an error
        RuntimeError: If the call fails or agent returns an error
        TimeoutError: If the invocation doesn't complete within timeout
        ValidationError: If response_model is provided and response doesn't match

    Examples:
        >>> # Untyped (returns dict)
        >>> result = await invoke("weather-agent", "get_forecast", {"city": "NYC"})
        >>> print(result["temperature"])

        >>> # Typed (returns WeatherResponse with IDE autocomplete)
        >>> result = await invoke("weather-agent", "get_forecast", {"city": "NYC"},
        ...                       response_model=WeatherResponse)
        >>> print(result.temperature)  # IDE knows this is WeatherResponse
    """
    # Convert Pydantic model to dict if needed
    if isinstance(payload, BaseModel):
        payload_dict = payload.model_dump()
    else:
        payload_dict = payload

    # Inherit context from current execution
    trace_id = _current_trace_id.get() or str(uuid.uuid4())
    # Child invocations should point to the current handler (invocation_id) as
    # their parent, not the current handler's own parent.
    parent_id = _current_invocation_id.get()

    # Build request body using typed model for API consistency
    invoke_request = InvokeFunctionRequest(
        agent_name=agent_name,
        function_name=function_name,
        payload=payload_dict,
        trace_id=trace_id,
        parent_id=parent_id,
        timeout_seconds=int(timeout),
    )
    invoke_body = invoke_request.model_dump(exclude_none=True)

    api_base_url = _get_api_base_url()
    auth_headers = _get_auth_headers()

    async with httpx.AsyncClient() as client:
        # Step 1: Start the invocation (returns immediately with invocation_id)
        response = await client.post(
            f"{api_base_url}/invoke",
            json=invoke_body,
            headers=auth_headers,
            timeout=10.0,
        )
        response.raise_for_status()
        start_result = response.json()

        invocation_id = start_result["invocation_id"]

        # Step 2: Poll for completion
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        while True:
            elapsed = loop.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Invocation {invocation_id} did not complete within {timeout}s"
                )

            # Check status
            status_response = await client.get(
                f"{api_base_url}/invoke/{invocation_id}",
                headers=auth_headers,
                timeout=10.0,
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status["status"] == "completed":
                result = status.get("result") or {}
                if response_model is not None:
                    return response_model.model_validate(result)
                return result

            if status["status"] == "error":
                raise RuntimeError(
                    f"Invoke failed: {status.get('error', 'Unknown error')}"
                )

            # Wait before next poll
            await asyncio.sleep(poll_interval)

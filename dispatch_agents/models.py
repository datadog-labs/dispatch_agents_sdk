"""Core models for the Dispatch SDK.

This module contains the fundamental data structures used across the entire
Dispatch ecosystem, including Message types for universal communication and Agent
for service registration and management.

Message Type Hierarchy (similar to LangChain's BaseMessage pattern):
- BaseMessage: Abstract base with common fields (uid, trace_id, sender_id, ts, payload)
- TopicMessage: For @on topic handlers (has 'topic' field)
- FunctionMessage: For @fn direct calls (has 'function_name' field)
- ScheduleMessage: For scheduled/cron triggers (has 'schedule_name' field)
- Message: Discriminated union type alias for routing
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


def get_now_utc() -> str:
    """Get the current UTC time in ISO8601 format."""
    return datetime.now(UTC).isoformat()


JsonSchema: TypeAlias = dict[str, Any]
"""A JSON Schema document, e.g. from Pydantic's model_json_schema()."""

# =============================================================================
# Feedback Types - Shared across backend, CLI, and SDK
# =============================================================================

FeedbackType: TypeAlias = Literal["bug", "feature_request", "general"]
"""Type of customer feedback submission."""

FeedbackSentiment: TypeAlias = Literal["positive", "negative"]
"""Thumbs up/down sentiment for feedback."""


class StrictBaseModel(BaseModel):
    """Base model with strict validation that forbids extra fields.

    All Dispatch models inherit from this to ensure API compatibility
    and catch typos in field names at validation time.
    """

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Message Types - Discriminated Union Pattern
# =============================================================================
# Following LangChain's BaseMessage pattern with Pydantic discriminated unions.
# The 'type' field acts as discriminator - Pydantic routes to correct subclass
# BEFORE validation, so strict validation (extra='forbid') works correctly.
# =============================================================================


class BaseMessage(StrictBaseModel):
    """Abstract base class for all dispatch messages.

    Similar to LangChain's BaseMessage pattern. Contains common fields
    shared across all message types. Should not be instantiated directly -
    use TopicMessage, FunctionMessage, or ScheduleMessage instead.

    | Field      | Type   | Required | Notes                                                                |
    | ---------- | ------ | -------- | -------------------------------------------------------------------- |
    | type       | string | always   | Discriminator: "topic", "function", or "schedule"                    |
    | payload    | object | always   | Business data, validated against registered input model              |
    | uid        | string | always   | Unique ID per message                                                |
    | trace_id   | string | always   | Groups related messages into a workflow/session                      |
    | sender_id  | string | always   | ID of the sending agent/tool                                         |
    | ts         | string | always   | ISO8601 timestamp of when message was created                        |
    | parent_id  | string | optional | UID of parent message for building trace trees (None for root events)|
    """

    type: str  # Abstract - subclasses override with Literal
    payload: Any
    uid: str
    trace_id: str
    sender_id: str
    ts: str
    parent_id: str | None = None


class TopicMessage(BaseMessage):
    """Message routed by topic subscription (@on decorator).

    Used for event-driven communication where agents subscribe to topics
    and receive messages published to those topics.
    """

    type: Literal["topic"] = "topic"
    topic: str = Field(description="Event topic for subscription-based routing")

    @classmethod
    def create(
        cls,
        topic: str,
        payload: Any,
        sender_id: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        _uid: str | None = None,
        _ts: str | None = None,
    ) -> "TopicMessage":
        """Create a new TopicMessage with auto-generated fields."""
        return cls(
            topic=topic,
            payload=payload,
            sender_id=sender_id,
            trace_id=trace_id or str(uuid.uuid4()),
            parent_id=parent_id,
            uid=_uid or str(uuid.uuid4()),
            ts=_ts or get_now_utc(),
        )


class FunctionMessage(BaseMessage):
    """Message for direct function invocation (@fn decorator).

    Used when one agent calls a function on another agent directly,
    bypassing topic-based routing.
    """

    type: Literal["function"] = "function"
    function_name: str = Field(description="Name of the @fn function to invoke")

    @classmethod
    def create(
        cls,
        function_name: str,
        payload: Any,
        sender_id: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        _uid: str | None = None,
        _ts: str | None = None,
    ) -> "FunctionMessage":
        """Create a new FunctionMessage with auto-generated fields."""
        return cls(
            type="function",
            function_name=function_name,
            payload=payload,
            sender_id=sender_id,
            trace_id=trace_id or str(uuid.uuid4()),
            parent_id=parent_id,
            uid=_uid or str(uuid.uuid4()),
            ts=_ts or get_now_utc(),
        )


class ScheduleMessage(BaseMessage):
    """Message triggered by a schedule/cron job.

    Used for time-based triggers where the system invokes an agent
    function based on a schedule configuration.
    """

    type: Literal["schedule"] = "schedule"
    schedule_name: str = Field(description="Schedule ID that triggered this invocation")
    function_name: str = Field(description="Name of the function being invoked")

    @classmethod
    def create(
        cls,
        schedule_name: str,
        function_name: str,
        payload: Any,
        sender_id: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        _uid: str | None = None,
        _ts: str | None = None,
    ) -> "ScheduleMessage":
        """Create a new ScheduleMessage with auto-generated fields."""
        return cls(
            type="schedule",
            schedule_name=schedule_name,
            function_name=function_name,
            payload=payload,
            sender_id=sender_id,
            trace_id=trace_id or str(uuid.uuid4()),
            parent_id=parent_id,
            uid=_uid or str(uuid.uuid4()),
            ts=_ts or get_now_utc(),
        )


class LLMCallMessage(BaseMessage):
    """Message representing an LLM inference call.

    Used to track LLM calls within a trace, enabling unified trace views
    that show LLM interactions alongside invocations and topic messages.

    The parent_id field can point to:
    - An invocation UID: The LLM call was made by this invocation
    - Another LLM call UID: The LLM call is a continuation (e.g., after tool execution)

    Children of this LLM call (invocations with parent_id pointing here) represent
    tool call results - invocations triggered by the LLM's tool_calls response.
    """

    type: Literal["llm_call"] = "llm_call"

    # LLM-specific fields
    model: str = Field(description="Model used (e.g., gpt-4o, claude-3-5-sonnet)")
    provider: str = Field(description="Provider (e.g., openai, anthropic)")
    messages: list[dict[str, Any]] = Field(description="Request messages sent to LLM")
    response: str | None = Field(default=None, description="LLM response text")
    finish_reason: str = Field(
        description="Completion reason: stop, tool_calls, length, etc."
    )

    # Usage metrics
    input_tokens: int = Field(description="Prompt token count")
    output_tokens: int = Field(description="Completion token count")
    cost_usd: float = Field(description="Calculated cost in USD")
    latency_ms: int = Field(description="Response time in milliseconds")

    # Optional fields
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="Tool definitions if function calling was used"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls from response"
    )
    variant_name: str | None = Field(
        default=None, description="A/B test variant if applicable"
    )

    @classmethod
    def create(
        cls,
        *,
        model: str,
        provider: str,
        messages: list[dict[str, Any]],
        finish_reason: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        sender_id: str,
        trace_id: str,
        parent_id: str | None = None,
        response: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        variant_name: str | None = None,
        _uid: str | None = None,
        _ts: str | None = None,
    ) -> "LLMCallMessage":
        """Create a new LLMCallMessage with auto-generated fields.

        Args:
            model: Model used for inference
            provider: LLM provider
            messages: Request messages
            finish_reason: Why the LLM stopped generating
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            latency_ms: Latency in milliseconds
            sender_id: Agent/caller that made this LLM call
            trace_id: Trace ID for correlation
            parent_id: Parent invocation or LLM call UID
            response: LLM response text
            tools: Tool definitions
            tool_calls: Tool calls from response
            variant_name: A/B test variant
            _uid: Override UID (auto-generated if not provided)
            _ts: Override timestamp
        """
        uid = _uid or str(uuid.uuid4())
        return cls(
            type="llm_call",
            model=model,
            provider=provider,
            messages=messages,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            sender_id=sender_id,
            trace_id=trace_id,
            parent_id=parent_id,
            response=response,
            tools=tools,
            tool_calls=tool_calls,
            variant_name=variant_name,
            # BaseMessage fields - payload is empty for LLM calls (data is in specific fields)
            payload={},
            uid=uid,
            ts=_ts or get_now_utc(),
        )


# Discriminated union - Pydantic routes by 'type' field before validation
# This means strict validation (extra='forbid') works correctly with subclass fields
Message = Annotated[
    TopicMessage | FunctionMessage | ScheduleMessage | LLMCallMessage,
    Field(discriminator="type"),
]


# =============================================================================
# Handler Response Payloads
# =============================================================================
# These are serialized into the gRPC InvokeResponse.result field.
# The proto's is_error bool tells the backend which type to deserialize to.
# =============================================================================


class SuccessPayload(StrictBaseModel):
    """Successful handler execution result.

    Serialized into InvokeResponse.result when is_error=False.
    """

    result: Any = Field(
        description="Handler return value (any JSON-serializable value)"
    )


class ErrorPayload(StrictBaseModel):
    """Failed handler execution result.

    Serialized into InvokeResponse.result when is_error=True.
    Contains structured error information for debugging and display.
    """

    error: str = Field(description="Error message")
    error_type: str = Field(
        description="Exception class name (e.g., 'ValidationError', 'ValueError')"
    )
    trace: str | None = Field(default=None, description="Full traceback for debugging")
    details: Any | None = Field(
        default=None,
        description="Additional error details (e.g., Pydantic validation errors list)",
    )


class AgentContainerStatus(StrEnum):
    BUILDING = auto()  # agent created/updated when codebuild starts
    DEPLOYING = auto()  # image built, ecs rolling update in progress
    DEPLOYED = auto()  # ip/dns alias available from ecs, ready to be health-checked
    ERROR = auto()  # ecs task failed to launch/retrying
    HEALTHY = auto()  # passed health check from in-memory registry
    UNHEALTHY = (
        auto()
    )  # deployed agent went from healthy->unhealthy or never passed health check
    DISABLED = (
        auto()
    )  # intentionally stopped/disabled, will not be considered for health check


class FunctionTrigger(StrictBaseModel):
    """Trigger configuration for an agent function.

    Defines how and when an agent function is invoked. Currently supports:
    - topic: Event-driven triggers from subscribed topics
    - callable: Direct function calls from other agents via invoke()
    - schedule: Time-based triggers (future)
    """

    type: Literal["topic", "schedule", "callable"] = Field(
        description="Trigger type: 'topic' for event-driven, 'callable' for direct invocation, 'schedule' for cron-based"
    )
    topic: str | None = Field(
        default=None,
        description="Topic name when type='topic'. Events published to this topic will invoke the function.",
    )
    function_name: str | None = Field(
        default=None,
        description="Function name when type='callable'. Other agents can invoke this function by name.",
    )
    # Future fields for schedule triggers:
    # cron_expression: str | None - Cron syntax for schedule
    # timezone: str | None - Timezone for schedule interpretation
    # enabled: bool - Whether schedule is active


class AgentFunction(StrictBaseModel):
    """Agent function with input/output schemas and triggers.

    Represents a handler function in an agent, including its schemas and
    the triggers that can invoke it (topics, schedules, etc.).
    """

    name: str = Field(description="Handler function name")
    description: str | None = Field(
        default=None, description="Handler docstring or description"
    )
    input_schema: JsonSchema = Field(
        description="JSON Schema for input payload validation"
    )
    output_schema: JsonSchema | None = Field(
        default=None, description="JSON Schema for output payload (if any)"
    )
    triggers: list[FunctionTrigger] = Field(
        default_factory=list, description="Triggers that invoke this function"
    )


class Agent(StrictBaseModel):
    """Agent registration and metadata model.

    Uses composite uid (org_id#namespace#name) as unique identifier.
    The name field stores the ECS-sanitized agent name.
    """

    # === Persistent fields (stored in DynamoDB) ===
    name: str = Field(
        description="ECS-sanitized agent name (used in uid composite key)"
    )
    org_id: str = Field(description="Organization ID for multi-tenancy")
    namespace: str = Field(description="Namespace for logical isolation within org")
    status: AgentContainerStatus = Field(
        default=AgentContainerStatus.BUILDING,
        description="Agent deployment/health status",
    )
    created_at: str = Field(description="ISO8601 timestamp when agent was created")
    last_updated: str = Field(
        description="ISO8601 timestamp when DB record was updated"
    )
    url: str | None = Field(default=None, description="Agent DNS alias endpoint URL")
    version: str | None = Field(
        default=None, description="Current deployed version from S3"
    )
    last_deployed: str | None = Field(
        default=None,
        description="ISO8601 timestamp of last successful deployment",
    )
    monthly_budget_usd: float | None = Field(
        default=None,
        description="Monthly LLM spend limit in USD. If set, inference requests "
        "are blocked when the agent exceeds this amount in the current month.",
    )

    # === Runtime fields (in-memory only, not persisted to DynamoDB) ===
    functions: list[AgentFunction] = Field(
        default_factory=list,
        description="Agent functions with their triggers, schemas, and metadata. "
        "Each function can have multiple triggers (topics, schedules, etc.).",
    )
    last_heartbeat: str | None = Field(
        default=None, description="ISO8601 timestamp of last heartbeat"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional runtime metadata"
    )

    @field_validator("name")
    def validate_name(cls, v):
        Agent._validate_identifier(v, "Agent name")
        Agent._sanitize_ecs_name(v, fallback=None)
        return v

    @field_validator("namespace")
    def validate_namespace(cls, v):
        Agent._validate_identifier(v, "Namespace")
        return v

    @classmethod
    def create(
        cls, name: str, functions: list[AgentFunction] | None = None, **kwargs
    ) -> "Agent":
        """Create a new Agent with ECS-sanitized name and timestamp."""
        sanitized_name = cls._sanitize_ecs_name(name)
        return cls(
            name=sanitized_name,
            functions=functions or [],
            created_at=get_now_utc(),
            last_updated=get_now_utc(),
            **kwargs,
        )

    def get_network_url(self, base_url: str | None = None) -> str:
        """Get the network URL for this agent."""
        if base_url:
            return f"{base_url.rstrip('/')}/{self.name}/dispatch"
        return f"http://{self.name}.trigger/dispatch"

    def handles_topic(self, topic: str) -> bool:
        """Check if this agent handles the given topic (supports wildcards)."""
        # Extract all topic triggers from functions
        for function in self.functions:
            for trigger in function.triggers:
                if trigger.type == "topic" and trigger.topic:
                    agent_topic = trigger.topic
                    if agent_topic.endswith("*"):
                        if topic.startswith(agent_topic[:-1]):
                            return True
                    elif agent_topic == topic:
                        return True
        return False

    @staticmethod
    def transform_topic_schemas_to_functions(
        topic_schemas: dict[str, dict[str, Any]],
    ) -> list[AgentFunction]:
        """Transform legacy topic_schemas dict to functions list.

        Used during agent registration to convert SDK-provided schemas
        into the new functions format.

        Args:
            topic_schemas: Dict mapping topic names to schema metadata.
                Each topic maps to a dict with keys: handler_name (str),
                handler_doc (str or None), input_schema (dict),
                output_schema (dict or None).

        Returns:
            List of AgentFunction objects, one per topic.
        """
        functions = []
        for topic, schema_metadata in topic_schemas.items():
            function = AgentFunction(
                name=schema_metadata.get("handler_name", "handler"),
                description=schema_metadata.get("handler_doc"),
                input_schema=schema_metadata.get("input_schema", {}),
                output_schema=schema_metadata.get("output_schema"),
                triggers=[FunctionTrigger(type="topic", topic=topic)],
            )
            functions.append(function)
        return functions

    @staticmethod
    def _validate_identifier(
        identifier: str, identifier_type: str = "identifier"
    ) -> None:
        """Validate that an identifier doesn't contain colon character.

        Colons are reserved as delimiters in task queue names to enable reliable parsing.

        Args:
            identifier: The string to validate (namespace, agent_name, etc.)
            identifier_type: Description of what's being validated (for error messages)

        Raises:
            ValueError: If identifier contains a colon
        """
        if ":" in identifier:
            raise ValueError(
                f"{identifier_type} cannot contain colon ':' character "
                f"(reserved as task queue delimiter): {identifier}"
            )

    @staticmethod
    def _sanitize_ecs_name(name: str, fallback: str | None = None) -> str:
        """Return a string valid for ECS names (family/service/container).
        If fallback is None, raise exception if name is invalid.
        Allowed characters are letters, numbers, hyphens, and underscores. Length must be 1-255.
        Any other character is replaced with '-'. If the result is empty, use the fallback.
        """
        sanitized = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in name)
        sanitized = sanitized.strip("-_")
        if sanitized:
            return sanitized[:255]
        elif fallback is not None:
            return Agent._sanitize_ecs_name(
                fallback, None
            )  # recurse to sanitize fallback
        else:
            raise ValueError(
                f"Name '{name}' cannot be sanitized, and no fallback provided (or fallback also could not be sanitized)."
            )

    @staticmethod
    def build_uid(org_id: str, namespace: str, agent_name: str) -> str:
        """Build composite agent UID from components.

        Args:
            org_id: Organization identifier
            namespace: Namespace identifier
            agent_name: Agent name (ECS-sanitized)

        Returns:
            Composite UID in format: org_id#namespace#agent_name

        Raises:
            ValueError: If any component contains '#' separator

        Example:
            >>> Agent.build_uid("org123", "default", "my-agent")
            "org123#default#my-agent"
        """
        if "#" in org_id or "#" in namespace or "#" in agent_name:
            raise ValueError(
                "Components cannot contain '#' separator. "
                f"org_id={org_id}, namespace={namespace}, agent_name={agent_name}"
            )
        return f"{org_id}#{namespace}#{agent_name}"

    @property
    def uid(self) -> str:
        """Computed unique identifier across all orgs and namespaces.

        This property is computed from org_id, namespace, and name.
        It is NOT stored in DynamoDB - only used as the primary key.

        Format: org_id#namespace#name

        Example:
            >>> agent = Agent(name="my-agent", org_id="org123", namespace="default", ...)
            >>> agent.uid
            "org123#default#my-agent"
        """
        return Agent.build_uid(self.org_id, self.namespace, self.name)


# Router API Models for compatibility across implementations
class PublishEventBody(StrictBaseModel):
    """Request body for publishing events to any dispatch router."""

    topic: str
    sender_id: str = "web-ui"
    payload: Any = Field(default_factory=dict)
    trace_id: str | None = None
    parent_id: str | None = None


class SubscriptionBody(StrictBaseModel):
    """Request body for agent subscription management."""

    topics: list[str]
    agent_name: str
    functions: list[AgentFunction] | None = None


class EventRequest(StrictBaseModel):
    """CLI router specific event request format."""

    payload: Any
    sender_id: str | None = "router"


class PublishResponse(StrictBaseModel):
    """Standard response for event publishing operations.

    When publishing to a topic, returns invocation IDs for all handlers triggered.
    Clients can poll these invocations to get results (similar to invoke() pattern).
    """

    message: str
    event_uid: str
    invocation_ids: list[str] = []  # Invocation IDs for handlers triggered
    handler_count: int = 0  # Number of handlers triggered


from dispatch_agents.invocation import InvocationStatus


class InvokeFunctionRequest(StrictBaseModel):
    """Request to invoke a function on an agent.

    This is the payload for POST /api/unstable/namespace/{namespace}/invoke
    Used by the SDK's invoke() function and must match backend/local router expectations.
    """

    agent_name: str = Field(description="Target agent name")
    function_name: str = Field(description="Function name to invoke")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Input payload for the function"
    )
    trace_id: str | None = Field(
        default=None, description="Optional trace ID for distributed tracing"
    )
    parent_id: str | None = Field(
        default=None, description="Optional parent span ID for distributed tracing"
    )
    timeout_seconds: int | None = Field(
        default=None,
        description="Optional timeout in seconds for the invocation. Defaults to 1 hour (3600s). Maximum is 24 hours (86400s).",
        ge=1,
        le=86400,
    )


class InvocationStatusResponse(StrictBaseModel):
    """Response from polling an invocation status.

    Returned by GET /api/unstable/namespace/{namespace}/invoke/{invocation_id}
    Contains the current status, agent/function info, and result (when completed).
    """

    invocation_id: str = Field(description="Unique invocation identifier")
    status: InvocationStatus = Field(description="Current invocation status")
    agent_name: str = Field(description="Target agent name")
    function_name: str = Field(description="Function name")
    trace_id: str = Field(description="Trace ID for distributed tracing")
    result: Any | None = Field(
        default=None, description="Result payload (when status is COMPLETED)"
    )
    error: str | None = Field(
        default=None, description="Error message (when status is ERROR)"
    )
    created_at: str = Field(
        description="ISO 8601 timestamp when invocation was created"
    )


class SubscriptionResponse(StrictBaseModel):
    """Standard response for subscription management operations."""

    message: str
    topics: list[str]
    agent_name: str
    subscribers: dict[str, int]


########################################################
# Memory Models
########################################################


class KVStoreRequest(StrictBaseModel):
    agent_name: str
    key: str
    value: str = Field(default="")


class SessionStoreRequest(StrictBaseModel):
    agent_name: str
    session_id: str
    session_data: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteResponse(StrictBaseModel):
    """Response from a memory write (add/delete) operation."""

    message: str


class KVGetResponse(StrictBaseModel):
    """Response from a long-term memory get operation."""

    value: str | None


class KVMemoryRecord(StrictBaseModel):
    """A single long-term memory record."""

    mem_key: str
    mem_value: str
    last_updated: str | None = None


class KVListResponse(StrictBaseModel):
    """Response from a long-term memory list operation."""

    agent_name: str
    memories: list[KVMemoryRecord]


class SessionGetResponse(StrictBaseModel):
    """Response from a short-term memory get operation."""

    session_data: dict[str, Any]

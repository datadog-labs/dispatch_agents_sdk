"""Public model definitions for the Dispatch Agents SDK.

This module is the source of truth for agent-facing models. Runtime and backend
wire contracts that are not intended for agent code live in
``dispatch_agents._internal.models``.

It includes:

- Base handler payload models
- Current invocation context models
- Agent invocation status models
- Eval/experiment payload models
- Runtime configuration models
- LLM response models
- Memory response models
- GitHub client response models

Quick start::

    from dispatch_agents.models import BasePayload

    class MyPayload(BasePayload):
        message: str
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType

from dispatch_agents._internal import config_validation as _config_validation

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    "JsonScalar | list[JsonValue] | dict[str, JsonValue]",
)
JsonArray: TypeAlias = list[JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]


class BasePayload(BaseModel):
    """Base class for all dispatch agent handler payloads.

    Handler input and output models should inherit from this class. It is a
    strict Pydantic model: unknown fields are rejected during validation so
    schema drift and payload typos fail early.

    Examples:
        >>> class MyEventPayload(BasePayload):
        ...     message: str
        ...     user_id: int
        ...
    """

    model_config = ConfigDict(extra="forbid")


class InvocationContext(BaseModel):
    """Identifiers for the currently running invocation."""

    trace_id: str
    invocation_id: str
    parent_id: str | None = None


class HandlerMetadata(BaseModel):
    """Serializable handler metadata for registration and introspection."""

    model_config = ConfigDict(extra="forbid")

    handler_name: str
    topics: list[str]
    input_schema: JsonObject
    output_schema: JsonObject | None
    handler_doc: str | None


class InvocationStatus(StrEnum):
    """Status of a direct function invocation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class InvocationResult(BaseModel):
    """Untyped result returned by a direct function invocation.

    The ``result`` attribute is the canonical API. Mapping-style access is kept
    for compatibility with existing agents that treated untyped invoke results
    as dictionaries.
    """

    result: JsonValue = None

    def __contains__(self, key: object) -> bool:
        if key == "result":
            return True
        return (
            isinstance(key, str)
            and isinstance(self.result, dict)
            and key in self.result
        )

    def __getitem__(self, key: str) -> JsonValue:
        if key == "result":
            return self.result
        if isinstance(self.result, dict):
            return self.result[key]
        raise KeyError(key)

    def get(self, key: str, default: JsonValue = None) -> JsonValue:
        if key == "result":
            return self.result
        if isinstance(self.result, dict):
            return self.result.get(key, default)
        return default


class EvalItem(BasePayload):
    """A single experiment case as sent to an invoker scorer.

    Fields mirror what the experiments runner tracks per case:

    - ``id`` — stable identifier the runner assigns per (experiment,
      item). Most single-item scorers can ignore it; batch scorers use
      it to map their dict-keyed return value back to specific rows.
    - ``input`` — payload the agent was invoked with.
    - ``expected`` — the human-supplied ground truth (may be ``None``).
    - ``output`` — whatever the agent returned.

    All non-id fields are typed ``Any`` because the runner doesn't
    enforce a schema on dataset cases — your agent decides the shape.
    JSON strings are auto-decoded before send, so dict fields land as
    ``dict``.

    Example::

        from dispatch_agents import fn
        from dispatch_agents.models import EvalItem

        class JudgeVerdict(BasePayload):
            score: int
            reason: str

        @fn()
        async def judge(case: EvalItem) -> JudgeVerdict:
            ...

    For batch scoring, see :class:`EvalBatch`.
    """

    id: str = ""
    input: Any = None
    expected: Any = None
    output: Any = None


class EvalBatch(BasePayload):
    """A batch of experiment cases for an invoker scorer.

    Use this when one scorer invocation should evaluate multiple cases
    at once (e.g., an LLM judge that scores N answers in a single
    prompt to amortize overhead). The runner groups cases into batches
    of ``InvokerConfig.batch_size`` before calling the scorer.

    The scorer is expected to return a ``dict[str, ...]`` keyed by
    :attr:`EvalItem.id` so the runner can map results back to rows.
    Missing keys are recorded as scorer errors for the affected
    cases; extra keys are dropped.

    Example::

        from dispatch_agents import fn
        from dispatch_agents.models import EvalBatch

        @fn()
        async def judge_batch(batch: EvalBatch) -> dict[str, dict]:
            results = {}
            for item in batch.items:
                results[item.id] = {"score": grade(item)}
            return results
    """

    items: list[EvalItem]


class MemoryWriteResponse(BaseModel):
    """Response from a memory write or delete operation."""

    model_config = ConfigDict(extra="forbid")

    message: str


class KVGetResponse(BaseModel):
    """Response from a long-term memory get operation."""

    model_config = ConfigDict(extra="forbid")

    value: str | None


class KVMemoryRecord(BaseModel):
    """A single long-term memory record returned by a list operation."""

    model_config = ConfigDict(extra="forbid")

    mem_key: str
    mem_value: str
    last_updated: str | None = None


class KVListResponse(BaseModel):
    """Response from a long-term memory list operation."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    memories: list[KVMemoryRecord]


class SessionGetResponse(BaseModel):
    """Response from a short-term memory get operation."""

    model_config = ConfigDict(extra="forbid")

    session_data: JsonObject = Field(default_factory=dict)


class LLMFunctionCall(BaseModel):
    """A function call within an LLM tool call."""

    name: str
    # "arguments" is a JSON-encoded string per the OpenAI chat completions API
    # (e.g. '{"location": "NYC"}'), not a collection. The singular concept is
    # "the arguments blob"; the plural name mirrors the upstream API field name.
    arguments: str


class LLMToolCall(BaseModel):
    """A tool call from the LLM response."""

    id: str
    type: str = "function"
    function: LLMFunctionCall


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""

    role: str  # system, user, assistant, tool
    content: str | list[JsonObject]
    name: str | None = None
    tool_call_id: str | None = None
    # Tool calls made by an assistant message (OpenAI-style call descriptors).
    # Same shape as ``LLMResponse.tool_calls`` so a response can be replayed as
    # the next request message without re-shaping.
    tool_calls: list[LLMToolCall] | None = None


class LLMResponse(BaseModel):
    """Response from LLM inference."""

    llm_call_id: str
    content: str | None
    tool_calls: list[LLMToolCall] | None
    finish_reason: str
    model: str
    provider: str
    variant_name: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class GitHubAppToken(BaseModel):
    """Short-lived GitHub App installation token returned by Dispatch."""

    token: str
    expires_at: datetime


class McpHttpServerConfig(BaseModel):
    """HTTP transport configuration for an MCP server."""

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


class MCPToolCallResult(BaseModel):
    """Result returned by an MCP tool call."""

    content: list[JsonObject] = Field(default_factory=list)
    is_error: bool = Field(default=False, alias="isError")

    model_config = ConfigDict(populate_by_name=True)


class MCPTool(BaseModel):
    """Tool definition returned by an MCP server."""

    name: str
    description: str | None = None
    input_schema: JsonObject = Field(default_factory=dict, alias="inputSchema")

    model_config = ConfigDict(populate_by_name=True)


class MCPListToolsResult(BaseModel):
    """Tools listed from an MCP server."""

    tools: list[MCPTool] = Field(default_factory=list)


class MCPResource(BaseModel):
    """Resource definition returned by an MCP server."""

    data: JsonObject = Field(default_factory=dict)


class MCPListResourcesResult(BaseModel):
    """Resources listed from an MCP server."""

    resources: list[MCPResource] = Field(default_factory=list)


class MCPReadResourceResult(BaseModel):
    """Resource contents returned by an MCP server."""

    contents: list[JsonObject] = Field(default_factory=list)


class MCPPrompt(BaseModel):
    """Prompt definition returned by an MCP server."""

    data: JsonObject = Field(default_factory=dict)


class MCPListPromptsResult(BaseModel):
    """Prompts listed from an MCP server."""

    prompts: list[MCPPrompt] = Field(default_factory=list)


class MCPGetPromptResult(BaseModel):
    """Prompt payload returned by an MCP server."""

    data: JsonObject = Field(default_factory=dict)


class VolumeMode(StrEnum):
    """Volume access mode for persistent storage."""

    READ_WRITE_MANY = "read_write_many"


class VolumeConfig(BaseModel):
    """Configuration for a persistent storage volume.

    Volumes provide persistent storage that survives container restarts
    and redeployments. Data is isolated per-agent.

    Example:
        volumes:
          - name: plans
            mountPath: /data/plans
            mode: read_write_many
    """

    name: str = Field(
        ...,
        description="Unique name for the volume (used for identification and cleanup)",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$",
    )
    mount_path: Annotated[
        str, AfterValidator(_config_validation.normalize_mount_path)
    ] = Field(
        ...,
        alias="mountPath",
        description="Path where the volume will be mounted inside the container (must be within /data)",
    )
    mode: VolumeMode = Field(
        ...,
        description="Access mode for the volume (required)",
    )

    model_config = {"populate_by_name": True}


class SecretConfig(BaseModel):
    """Configuration for a secret to be injected as an environment variable.

    Secrets are retrieved from the secrets manager and injected into the
    container as environment variables at runtime.

    Example:
        secrets:
          - name: OPENAI_API_KEY
            secret_id: /shared/openai-api-key
    """

    name: str = Field(
        ...,
        description="Environment variable name for the secret",
        min_length=1,
    )
    secret_id: str = Field(
        ...,
        description="Path to the secret in secrets manager",
        min_length=1,
    )


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server to connect to.

    Example:
        mcp_servers:
          - server: datadog
    """

    server: str = Field(
        ...,
        description="MCP server installation name from the registry",
        min_length=1,
    )


class ResourceLimits(BaseModel):
    """CPU and memory limits for a container.

    CPU is specified in Kubernetes format:
    - Millicores: "250m", "500m", "1000m"
    - Cores: "0.25", "0.5", "1", "2"

    Memory is specified in Kubernetes format:
    - Mebibytes: "512Mi", "1024Mi"
    - Gibibytes: "1Gi", "2Gi"

    Example:
        limits:
          cpu: "500m"
          memory: "1Gi"
    """

    cpu: Annotated[str, AfterValidator(_config_validation.validate_cpu)] = Field(
        default="250m",
        description="CPU (e.g., '250m', '500m', '1', '2')",
    )
    memory: Annotated[str, AfterValidator(_config_validation.validate_memory)] = Field(
        default="2Gi",
        description="Memory (e.g., '512Mi', '1Gi', '2Gi')",
    )

    @model_validator(mode="after")
    def _validate_combination(self) -> ResourceLimits:
        return _config_validation.validate_resource_limits(self)


class ResourceConfig(BaseModel):
    """Configuration for agent container resources.

    Resources are expressed as limits.

    Example:
        resources:
          limits:
            cpu: "500m"
            memory: "1Gi"
    """

    limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Resource limits (CPU and memory)",
    )


class DomainSelector(BaseModel):
    """A single domain selector -- exactly one of match_name or match_pattern.

    match_name is an exact FQDN (e.g. api.openai.com).
    match_pattern is a wildcard prefix (e.g. ``*.github.com``).

    Serialises with camelCase aliases (matchName / matchPattern) to match the
    downstream Cilium FQDN selector API.
    """

    match_name: str | None = Field(
        default=None,
        description="Exact FQDN to allow. Must match the entire domain name exactly "
        "(e.g. 'api.openai.com' matches only 'api.openai.com').",
    )
    match_pattern: str | None = Field(
        default=None,
        description="Wildcard pattern to allow. Uses '*.domain.com' syntax to match "
        "any subdomain of the specified domain (e.g. '*.github.com' matches "
        "'api.github.com' and 'raw.github.com' but not 'github.com' itself).",
    )

    @model_validator(mode="after")
    def _validate_exactly_one_field(self) -> DomainSelector:
        return _config_validation.validate_domain_selector(self)

    model_config = {"extra": "forbid"}


class EgressConfig(BaseModel):
    """Configuration for network egress allow list.

    Domains are specified as objects with either matchName (exact FQDN)
    or matchPattern (wildcard prefix). This is a subset of the
    downstream Cilium FQDN selector API.

    Example::

        network:
          egress:
            allow_domains:
              - match_name: api.openai.com
              - match_pattern: "*.github.com"
    """

    allow_domains: Annotated[
        list[DomainSelector],
        AfterValidator(_config_validation.validate_allow_domains),
    ] = Field(
        default_factory=list,
        description="Domains allowed for egress as Cilium FQDN selectors.",
    )


class NetworkConfig(BaseModel):
    """Network configuration for an agent.

    When present in dispatch.yaml, CiliumNetworkPolicies are created to
    restrict the agent's outbound traffic to platform services and any
    listed allow_domains.  When absent, all egress is unrestricted.

    Example::

        network:
          egress:
            allow_domains:
              - match_name: api.openai.com
              - match_pattern: "*.github.com"
    """

    egress: EgressConfig = Field(default_factory=EgressConfig)


class DispatchConfig(BaseModel):
    """Configuration model for dispatch.yaml files.

    This model defines the complete schema for agent deployment configuration.
    It supports validation, serialization, and provides clear documentation
    for all configuration options.

    Example dispatch.yaml::

        namespace: skunkworks
        agent_name: my-agent
        entrypoint: agent.py
        base_image: python:3.13-slim
        env:
          LOG_LEVEL: debug
          MY_APP_MODE: production
        volumes:
          - name: data
            mountPath: /data
            mode: read_write_many
        secrets:
          - name: OPENAI_API_KEY
            secret_id: /shared/openai-api-key
        resources:
          limits:
            cpu: "500m"
            memory: "1Gi"
    """

    namespace: str | None = Field(
        default=None,
        description="Namespace for agent deployment (required for deployment)",
    )
    agent_name: str | None = Field(
        default=None,
        description="Unique name for the agent",
    )
    entrypoint: str | None = Field(
        default=None,
        description="Python file containing agent handlers (default: agent.py)",
    )
    base_image: str | None = Field(
        default=None,
        description="Base Docker image for the agent container",
    )
    system_packages: list[str] | None = Field(
        default=None,
        description="Additional system packages to install (apt packages)",
    )
    local_dependencies: dict[str, str] | None = Field(
        default=None,
        description="Local path dependencies to bundle (name -> path mapping)",
    )
    env: Annotated[
        dict[str, str] | None,
        BeforeValidator(_config_validation.check_env_values_are_strings),
        AfterValidator(_config_validation.validate_reserved_env),
    ] = Field(
        default=None,
        description="Plain environment variables to inject into the container (non-secret)",
    )

    vars: Annotated[
        JsonObject | None,
        BeforeValidator(_config_validation.validate_vars),
    ] = Field(
        default=None,
        description="Configuration variables accessible at runtime via dispatch_agents.config.vars. "
        "Unlike env, these are NOT injected as environment variables. "
        "Supports any YAML-serializable type. Use {value: <any>, description: <str>} "
        "to attach descriptions for the UI.",
    )

    secrets: list[SecretConfig] | None = Field(
        default=None,
        description="Secrets to inject as environment variables",
    )
    volumes: list[VolumeConfig] | None = Field(
        default=None,
        description="Persistent storage volumes to mount",
    )
    mcp_servers: list[MCPServerConfig] | None = Field(
        default=None,
        description="MCP servers to connect to from the registry",
    )
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig,
        description="Container resource limits (CPU and memory)",
    )
    network: NetworkConfig | None = Field(
        default=None,
        description="Network egress restrictions. When set, CiliumNetworkPolicies restrict outbound traffic.",
    )
    llm_instrument: bool = Field(
        default=True,
        description=(
            "Route LLM calls through the Dispatch sidecar proxy for tracing and "
            "cost tracking. Set false to call providers directly with your own keys."
        ),
    )
    log_level: str | None = Field(
        default=None,
        description=(
            "SDK log verbosity: DEBUG, INFO, WARNING, or ERROR (case-insensitive). "
            "When unset, the SDK logs at WARNING."
        ),
    )

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid log_level: {value!r}. "
                f"Must be one of: {', '.join(sorted(allowed))}."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_env_secrets_no_overlap(self) -> DispatchConfig:
        return _config_validation.validate_env_secret_overlap(self)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for YAML serialization.

        Excludes None values, drops empty collections, and converts nested
        models to plain dicts with canonical key ordering. ``resources`` is
        always included because it carries defaults.
        """
        result: dict[str, Any] = {}

        if self.namespace is not None:
            result["namespace"] = self.namespace
        if self.agent_name is not None:
            result["agent_name"] = self.agent_name
        if self.entrypoint is not None:
            result["entrypoint"] = self.entrypoint
        if self.base_image is not None:
            result["base_image"] = self.base_image
        if self.system_packages:
            result["system_packages"] = self.system_packages
        if self.local_dependencies:
            result["local_dependencies"] = self.local_dependencies
        if self.env:
            result["env"] = dict(self.env)
        if self.vars:
            result["vars"] = dict(self.vars)
        if self.secrets:
            result["secrets"] = [
                {"name": s.name, "secret_id": s.secret_id} for s in self.secrets
            ]
        if self.mcp_servers:
            result["mcp_servers"] = [{"server": m.server} for m in self.mcp_servers]
        if self.volumes:
            result["volumes"] = [
                {"name": v.name, "mountPath": v.mount_path, "mode": v.mode.value}
                for v in self.volumes
            ]
        # Always include resources since it has defaults.
        result["resources"] = {
            "limits": {
                "cpu": self.resources.limits.cpu,
                "memory": self.resources.limits.memory,
            }
        }

        if self.network is not None:
            result["network"] = {
                "egress": {
                    "allow_domains": [
                        d.model_dump(exclude_none=True)
                        for d in self.network.egress.allow_domains
                    ],
                }
            }

        # Only serialize when the author opted out of the default (instrumented).
        if not self.llm_instrument:
            result["llm_instrument"] = False

        if self.log_level is not None:
            result["log_level"] = self.log_level

        return result

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


__all__ = [
    "BasePayload",
    "DispatchConfig",
    "DomainSelector",
    "EgressConfig",
    "EvalBatch",
    "EvalItem",
    "GitHubAppToken",
    "HandlerMetadata",
    "InvocationContext",
    "InvocationResult",
    "InvocationStatus",
    "JsonArray",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "KVGetResponse",
    "KVListResponse",
    "KVMemoryRecord",
    "LLMFunctionCall",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "MemoryWriteResponse",
    "SessionGetResponse",
    "MCPServerConfig",
    "MCPGetPromptResult",
    "MCPListPromptsResult",
    "MCPListResourcesResult",
    "MCPListToolsResult",
    "MCPPrompt",
    "MCPReadResourceResult",
    "MCPResource",
    "MCPTool",
    "MCPToolCallResult",
    "McpHttpServerConfig",
    "NetworkConfig",
    "ResourceConfig",
    "ResourceLimits",
    "SecretConfig",
    "VolumeConfig",
    "VolumeMode",
]

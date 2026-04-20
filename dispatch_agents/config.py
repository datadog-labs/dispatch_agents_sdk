"""Configuration models for dispatch.yaml files.

This module defines the schema for agent deployment configuration,
shared between CLI and backend.
"""

import os
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from dispatch_agents.resources import _parse_cpu, _parse_memory

# Env var names managed by the platform — users must not override these.
RESERVED_ENV_VARS: frozenset[str] = frozenset(
    {
        "BACKEND_URL",
        "DISPATCH_API_KEY",
        "DISPATCH_NAMESPACE",
        "DISPATCH_AGENT_NAME",
        "DISPATCH_AGENT_VERSION",
        "MCP_CONFIG_JSON",
        "MCP_GATEWAY_URL",
    }
)

# Domain validation pattern for egress allow list.
# Accepts exact FQDNs (api.openai.com) and wildcard prefixes (*.github.com).
# Rejects URLs with schemes/ports/paths, IP addresses, and bare wildcards.
_DOMAIN_PATTERN = re.compile(
    r"^(\*\.)?([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


class VolumeMode(StrEnum):
    """Volume access mode for persistent storage.

    Currently only read_write_many is supported, but this enum allows
    for future expansion to other modes like read_only, read_write_once, etc.
    """

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
    mount_path: str = Field(
        ...,
        alias="mountPath",
        description="Path where the volume will be mounted inside the container (must be within /data)",
    )
    mode: VolumeMode = Field(
        ...,
        description="Access mode for the volume (required)",
    )

    @field_validator("mount_path")
    @classmethod
    def _validate_mount_path(cls, v: str) -> str:
        """Ensure mount path is within /data directory."""
        if not v.startswith("/data"):
            raise ValueError(
                f"mountPath must be within /data directory, got: {v}. "
                "Example: /data/plans or /data"
            )
        # Normalize path (remove trailing slashes, etc.)
        return os.path.normpath(v)

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


def _get_valid_memory_for_cpu(cpu_units: int) -> list[int] | None:
    """Get valid memory values (in MB) for a given CPU value.

    Returns None if the CPU value is not valid.

    Valid CPU/Memory combinations for AWS ECS Fargate:
    https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html
    """
    # CPU units: 1024 = 1 vCPU, Memory: MB
    valid_combinations: dict[int, list[int]] = {
        256: [512, 1024, 2048],
        512: [1024, 2048, 3072, 4096],
        1024: [2048, 3072, 4096, 5120, 6144, 7168, 8192],
        2048: [
            4096,
            5120,
            6144,
            7168,
            8192,
            9216,
            10240,
            11264,
            12288,
            13312,
            14336,
            15360,
            16384,
        ],
        4096: [
            8192,
            9216,
            10240,
            11264,
            12288,
            13312,
            14336,
            15360,
            16384,
            17408,
            18432,
            19456,
            20480,
            21504,
            22528,
            23552,
            24576,
            25600,
            26624,
            27648,
            28672,
            29696,
            30720,
        ],
        8192: list(range(16384, 61441, 4096)),  # 16GB to 60GB in 4GB increments
        16384: list(range(32768, 122881, 8192)),  # 32GB to 120GB in 8GB increments
    }
    return valid_combinations.get(cpu_units)


def _get_valid_cpu_values() -> list[int]:
    """Get list of valid CPU values (in internal units where 1024 = 1 vCPU)."""
    return [256, 512, 1024, 2048, 4096, 8192, 16384]


def _format_cpu(cpu_units: int) -> str:
    """Format internal CPU units to Kubernetes-style string.

    Examples:
        256 -> "250m"
        512 -> "500m"
        1024 -> "1"
        2048 -> "2"
    """
    if cpu_units >= 1024 and cpu_units % 1024 == 0:
        # Whole cores
        return str(cpu_units // 1024)
    else:
        # Millicores
        millicores = int(cpu_units * 1000 / 1024)
        return f"{millicores}m"


def _format_memory(memory_mb: int) -> str:
    """Format memory in MB to Kubernetes-style string.

    Examples:
        512 -> "512Mi"
        1024 -> "1Gi"
        2048 -> "2Gi"
    """
    if memory_mb >= 1024 and memory_mb % 1024 == 0:
        # Use Gi for whole gibibytes
        return f"{memory_mb // 1024}Gi"
    else:
        return f"{memory_mb}Mi"


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

    cpu: str = Field(
        default="250m",
        description="CPU (e.g., '250m', '500m', '1', '2')",
    )
    memory: str = Field(
        default="2Gi",
        description="Memory (e.g., '512Mi', '1Gi', '2Gi')",
    )

    @field_validator("cpu")
    @classmethod
    def _validate_cpu(cls, v: str) -> str:
        """Parse and validate CPU value."""
        try:
            cpu_units = _parse_cpu(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid CPU format '{v}': {e}") from e

        valid_cpu_values = _get_valid_cpu_values()
        if cpu_units not in valid_cpu_values:
            valid_cpus = [_format_cpu(c) for c in sorted(valid_cpu_values)]
            raise ValueError(
                f"Invalid CPU value '{v}' ({cpu_units} units). "
                f"Must be one of: {valid_cpus}"
            )
        return v

    @field_validator("memory")
    @classmethod
    def _validate_memory(cls, v: str) -> str:
        """Parse and validate memory value."""
        try:
            memory_mb = _parse_memory(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid memory format '{v}': {e}") from e

        # Minimum memory limit: 256 MB (required for sidecar overhead)
        min_memory_mb = 256
        if memory_mb < min_memory_mb:
            raise ValueError(
                f"Memory value '{v}' ({memory_mb} MB) is below minimum allowed "
                f"of {_format_memory(min_memory_mb)}."
            )

        # Maximum memory limit: 16GB (16384 MB)
        max_memory_mb = 16384
        if memory_mb > max_memory_mb:
            raise ValueError(
                f"Memory value '{v}' ({memory_mb} MB) exceeds maximum allowed "
                f"of {_format_memory(max_memory_mb)}. "
                "Contact support if you need more resources."
            )
        return v

    @model_validator(mode="after")
    def _validate_combination(self) -> "ResourceLimits":
        """Validate that CPU and memory form a valid combination."""
        cpu_units = _parse_cpu(self.cpu)
        memory_mb = _parse_memory(self.memory)

        valid_memory_values = _get_valid_memory_for_cpu(cpu_units) or []
        if memory_mb not in valid_memory_values:
            valid_memory_strs = [_format_memory(m) for m in valid_memory_values]
            raise ValueError(
                f"Invalid resource combination: CPU {self.cpu} with memory {self.memory}. "
                f"For CPU {self.cpu}, valid memory values are: {valid_memory_strs}"
            )
        return self


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
    match_pattern is a wildcard prefix (e.g. *.github.com).

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
    def _validate_exactly_one_field(self) -> "DomainSelector":
        if self.match_name and self.match_pattern:
            raise ValueError(
                "Exactly one of match_name or match_pattern must be set, not both"
            )
        if not self.match_name and not self.match_pattern:
            raise ValueError("Exactly one of match_name or match_pattern must be set")
        domain = self.match_name or self.match_pattern or ""
        if not _DOMAIN_PATTERN.match(domain):
            raise ValueError(
                f"Invalid domain '{domain}'. "
                "Must be an exact FQDN (e.g., api.openai.com) "
                "or wildcard prefix (e.g., *.github.com). "
                "URL schemes, ports, paths, IP addresses, "
                "and bare wildcards are not allowed."
            )
        return self

    model_config = {"extra": "forbid"}


class EgressConfig(BaseModel):
    """Configuration for network egress allow list.

    Domains are specified as objects with either matchName (exact FQDN)
    or matchPattern (wildcard prefix). This is a subset of the
    downstream Cilium FQDN selector API.

    Example:
        network:
          egress:
            allow_domains:
              - match_name: api.openai.com
              - match_pattern: "*.github.com"
    """

    allow_domains: list[DomainSelector] = Field(
        default_factory=list,
        description="Domains allowed for egress as Cilium FQDN selectors.",
    )

    @field_validator("allow_domains")
    @classmethod
    def _validate_allow_domains(cls, v: list[DomainSelector]) -> list[DomainSelector]:
        if len(v) > 50:
            raise ValueError(
                f"allow_domains cannot have more than 50 entries, got {len(v)}"
            )
        return v


class NetworkConfig(BaseModel):
    """Network configuration for an agent.

    When present in dispatch.yaml, CiliumNetworkPolicies are created to
    restrict the agent's outbound traffic to platform services and any
    listed allow_domains.  When absent, all egress is unrestricted.

    Example:
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

    Example dispatch.yaml:
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
    env: dict[str, str] | None = Field(
        default=None,
        description="Plain environment variables to inject into the container (non-secret)",
    )

    @field_validator("env", mode="before")
    @classmethod
    def _check_env_values_are_strings(cls, v: dict | None) -> dict[str, str] | None:
        """Ensure all env values are strings.

        YAML parses unquoted ``false`` as bool, ``123`` as int, etc.
        Since env vars are always strings, require explicit quoting.
        """
        if not v or not isinstance(v, dict):
            return v
        non_string = {
            k: type(val).__name__ for k, val in v.items() if not isinstance(val, str)
        }
        if non_string:
            examples = ", ".join(
                f'{k} (got {t}, wrap in quotes: "{v[k]}")'
                for k, t in sorted(non_string.items())
            )
            raise ValueError(f"All env values must be strings. {examples}")
        return v

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

    @field_validator("env")
    @classmethod
    def _validate_env(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Reject env vars that collide with platform-managed names."""
        if not v:
            return v
        collisions = RESERVED_ENV_VARS & v.keys()
        if collisions:
            raise ValueError(
                f"Cannot set reserved environment variable(s): "
                f"{', '.join(sorted(collisions))}. "
                "These are managed by the Dispatch platform."
            )
        return v

    @model_validator(mode="after")
    def _validate_env_secrets_no_overlap(self) -> "DispatchConfig":
        """Ensure no env var name also appears as a secret name."""
        if not self.env or not self.secrets:
            return self
        env_names = set(self.env.keys())
        secret_names = {s.name for s in self.secrets}
        overlap = env_names & secret_names
        if overlap:
            raise ValueError(
                f"Environment variable(s) {', '.join(sorted(overlap))} "
                "defined in both 'env' and 'secrets'. "
                "Use 'secrets' for sensitive values or 'env' for non-secret values, not both."
            )
        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for YAML serialization.

        Excludes None values and converts nested models to dicts.
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
        # Always include resources since it has defaults
        limits_dict: dict[str, str] = {
            "cpu": self.resources.limits.cpu,
            "memory": self.resources.limits.memory,
        }
        result["resources"] = {"limits": limits_dict}

        if self.network is not None:
            result["network"] = {
                "egress": {
                    "allow_domains": [
                        d.model_dump(exclude_none=True)
                        for d in self.network.egress.allow_domains
                    ],
                }
            }

        return result

    model_config = {"populate_by_name": True}

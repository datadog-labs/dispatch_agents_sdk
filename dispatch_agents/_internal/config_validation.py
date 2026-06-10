"""Validation helpers for public configuration models."""

from __future__ import annotations

import os
import re
from typing import Any, TypeAlias, TypeGuard, TypeVar

_T = TypeVar("_T")

JsonScalar: TypeAlias = str | int | float | bool | None
JsonArray: TypeAlias = list["JsonValue"]
JsonObject: TypeAlias = dict[str, "JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonArray | JsonObject

# Every environment variable Dispatch injects into agent containers is prefixed
# with ``DISPATCH_``. Reserving the whole prefix (rather than a hand-maintained
# list) means new platform vars are automatically protected from user collisions.
# Behavioral toggles that used to be overridable here (log level, LLM
# instrumentation) are now first-class ``dispatch.yaml`` fields instead.
RESERVED_ENV_PREFIX = "DISPATCH_"

_DESCRIBED_KEYS = {"value", "description"}
_DOMAIN_PATTERN = re.compile(
    r"^(\*\.)?([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


def _parse_cpu(value: str | int | float) -> int:
    """Parse a Kubernetes CPU value to internal units."""
    if isinstance(value, int | float):
        return int(value * 1024)

    value_str = str(value).strip().lower()
    if value_str.endswith("m"):
        millicores = int(value_str[:-1])
        return int(millicores * 1024 / 1000)
    return int(float(value_str) * 1024)


def _parse_memory(value: str | int) -> int:
    """Parse a Kubernetes memory value to MB."""
    if isinstance(value, int):
        return value

    value_str = str(value).strip()
    if value_str.endswith("Gi"):
        return int(float(value_str[:-2]) * 1024)
    if value_str.endswith("Mi"):
        return int(float(value_str[:-2]))
    if value_str.endswith("G"):
        return int(float(value_str[:-1]) * 1000)
    if value_str.endswith("M"):
        return int(float(value_str[:-1]))
    return int(float(value_str))


def _format_cpu(cpu_units: int) -> str:
    """Format internal CPU units to a human-readable string."""
    if cpu_units % 1024 == 0:
        return str(cpu_units // 1024)
    return f"{cpu_units * 1000 // 1024}m"


def _format_memory(memory_mb: int) -> str:
    """Format memory in MB to a human-readable string."""
    if memory_mb % 1024 == 0:
        return f"{memory_mb // 1024}Gi"
    return f"{memory_mb}Mi"


def normalize_mount_path(value: str) -> str:
    """Validate and normalize a volume mount path."""
    if not value.startswith("/data"):
        raise ValueError(
            f"mountPath must be within /data directory, got: {value}. "
            "Example: /data/plans or /data"
        )
    return os.path.normpath(value)


_VALID_CPU_UNITS = [256, 512, 1024, 2048, 4096, 8192, 16384]
_VALID_CPU_STRINGS = [
    "250m",
    "500m",
    "1000m",
    "1",
    "2",
    "4",
    "8",
    "16",
    "0.25",
    "0.5",
    "1.0",
    "2.0",
    "4.0",
    "8.0",
    "16.0",
]
_MAX_MEMORY_MB = 16 * 1024  # 16Gi
_MIN_MEMORY_MB = 256  # 256Mi


def validate_cpu(value: str) -> str:
    """Validate a CPU limit value."""
    if not value:
        raise ValueError(f"Invalid CPU format: {value!r}. Must be a valid CPU value.")
    try:
        cpu_units = _parse_cpu(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid CPU format: {value!r}. Must be a valid CPU value."
        ) from None
    if cpu_units <= 0:
        raise ValueError(f"Invalid CPU format: {value!r}. Must be a valid CPU value.")
    valid = _get_valid_cpu_values()
    if cpu_units not in valid:
        valid_strs = ", ".join(_format_cpu(u) for u in valid)
        raise ValueError(f"Invalid CPU value: {value}. Must be one of: {valid_strs}")
    return value


def validate_memory(value: str) -> str:
    """Validate a memory limit value."""
    if not value:
        raise ValueError(
            f"Invalid memory format: {value!r}. Must be a valid memory value."
        )
    try:
        memory_mb = _parse_memory(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid memory format: {value!r}. Must be a valid memory value."
        ) from None
    if memory_mb <= 0:
        raise ValueError(
            f"Invalid memory format: {value!r}. Must be a valid memory value."
        )
    if memory_mb < _MIN_MEMORY_MB:
        raise ValueError(
            f"Memory {value} is below minimum allowed (256Mi). Must be at least 256Mi."
        )
    if memory_mb > _MAX_MEMORY_MB:
        raise ValueError(
            f"Memory {value} exceeds maximum allowed (16Gi). Must be at most 16Gi."
        )
    return value


def validate_resource_limits(value: _T) -> _T:
    """Validate CPU and memory limits as a Fargate-compatible pair."""
    v: Any = value
    cpu_units = _parse_cpu(v.cpu)
    memory_mb = _parse_memory(v.memory)
    valid_memory = _get_valid_memory_for_cpu(cpu_units)
    if valid_memory is not None and memory_mb not in valid_memory:
        valid_values = ", ".join(_format_memory(m) for m in valid_memory)
        raise ValueError(
            f"Invalid resource combination: CPU {v.cpu} with "
            f"memory {v.memory}. "
            f"For this CPU, valid memory values are: {valid_values}"
        )
    return value


def check_env_values_are_strings(value: object) -> object:
    """Validate env values are strings before Pydantic coercion."""
    if value is None:
        return value
    if not isinstance(value, dict):
        raise ValueError("env must be a mapping of string keys to string values")
    for key, env_value in value.items():
        if not isinstance(key, str) or not isinstance(env_value, str):
            raise ValueError("env must contain only string keys and string values")
    return value


def reserved_env_keys(value: dict[str, str] | None) -> list[str]:
    """Return sorted user env keys that collide with the platform-reserved prefix.

    A key is reserved when it starts with ``DISPATCH_``. Behavioral toggles are
    configured via dedicated ``dispatch.yaml`` fields, not env overrides.
    """
    if not value:
        return []
    return sorted(key for key in value if key.startswith(RESERVED_ENV_PREFIX))


def validate_reserved_env(value: dict[str, str] | None) -> dict[str, str] | None:
    """Reject environment variables reserved by Dispatch."""
    if value is None:
        return value
    reserved = reserved_env_keys(value)
    if reserved:
        joined = ", ".join(reserved)
        raise ValueError(
            f"Environment variables starting with {RESERVED_ENV_PREFIX!r} are "
            f"reserved by Dispatch and cannot be set: {joined}."
        )
    return value


def validate_vars(value: object) -> JsonObject | None:
    """Validate runtime vars from dispatch.yaml.

    Each var value may be:
    - A scalar (str, int, float, bool, None)
    - A list
    - A described-var dict with only ``value`` and/or ``description`` keys.
      Plain dicts (without a ``value`` key) are rejected to prevent accidental
      collisions with the described-var schema.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("vars must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValueError("vars keys must be strings")
    for key, var_value in value.items():
        if isinstance(var_value, dict):
            if "value" not in var_value and "description" not in var_value:
                raise ValueError(
                    f"var {key!r}: dict value without a 'value' key is not allowed. "
                    "Nest structured data under 'value' instead."
                )
            if "value" not in var_value and "description" in var_value:
                raise ValueError(
                    f"var {key!r}: dict value without a 'value' key is not allowed. "
                    "A described-var must include a 'value' key."
                )
            unexpected = set(var_value) - _DESCRIBED_KEYS
            if unexpected:
                raise ValueError(
                    f"var {key!r}: dict has unexpected keys: "
                    f"{sorted(unexpected)}. Only 'value' and 'description' are allowed."
                )
    return value


def validate_domain_selector(value: _T) -> _T:
    """Validate that a domain selector has exactly one selector field."""
    v: Any = value
    match_name = v.match_name
    match_pattern = v.match_pattern
    if bool(match_name) and bool(match_pattern):
        raise ValueError("Specify match_name or match_pattern, not both.")
    if not bool(match_name) and not bool(match_pattern):
        raise ValueError("Exactly one of match_name or match_pattern must be set")
    domain = match_name or match_pattern
    if not isinstance(domain, str) or not _DOMAIN_PATTERN.match(domain):
        raise ValueError(f"Invalid domain selector: {domain}")
    return value


_MAX_ALLOW_DOMAINS = 50


def validate_allow_domains(value: list[object]) -> list[object]:
    """Validate a list of domain selectors."""
    if len(value) > _MAX_ALLOW_DOMAINS:
        raise ValueError(
            f"allow_domains cannot have more than {_MAX_ALLOW_DOMAINS} entries "
            f"(got {len(value)})."
        )
    return value


def validate_env_secret_overlap(value: _T) -> _T:
    """Reject duplicate names in env and secrets."""
    v: Any = value
    env = v.env
    secrets = v.secrets
    if not env or not secrets:
        return value
    secret_names = {secret.name for secret in secrets}
    overlap = sorted(set(env) & secret_names)
    if overlap:
        joined = ", ".join(overlap)
        raise ValueError(
            f"Variables cannot be defined in both env and secrets: {joined}"
        )
    return value


def is_described_var(value: JsonValue) -> TypeGuard[dict[str, JsonValue]]:
    """Return whether a var uses the described-var object shape."""
    return (
        isinstance(value, dict)
        and set(value) == _DESCRIBED_KEYS
        and isinstance(value.get("description"), str)
    )


def unwrap_described_vars(vars_data: JsonObject) -> JsonObject:
    """Return vars with described values unwrapped."""
    unwrapped: JsonObject = {}
    for key, value in vars_data.items():
        if is_described_var(value):
            unwrapped[key] = value["value"]
        else:
            unwrapped[key] = value
    return unwrapped


def extract_var_descriptions(vars_data: JsonObject) -> dict[str, str]:
    """Return descriptions for described runtime vars."""
    descriptions: dict[str, str] = {}
    for key, value in vars_data.items():
        if is_described_var(value) and isinstance(value["description"], str):
            descriptions[key] = value["description"]
    return descriptions


def _get_valid_memory_for_cpu(cpu_units: int) -> list[int] | None:
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
        8192: [
            16384,
            20480,
            24576,
            28672,
            32768,
            36864,
            40960,
            45056,
            49152,
            53248,
            57344,
            61440,
        ],
        16384: [
            32768,
            40960,
            49152,
            57344,
            65536,
            73728,
            81920,
            90112,
            98304,
            106496,
            114688,
            122880,
        ],
    }
    return valid_combinations.get(cpu_units)


def _get_valid_cpu_values() -> list[int]:
    """Return the list of valid CPU unit values."""
    return [256, 512, 1024, 2048, 4096, 8192, 16384]

"""Deprecated public compatibility aliases.

This module is the single cleanup point for old public import paths that still
have a current public owner. Do not add aliases that point to ``_internal``.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from types import ModuleType
from typing import Any

AliasTargets = dict[str, str]

_STATIC_ALIASES: dict[str, AliasTargets] = {
    "dispatch_agents": {
        "HandlerMetadata": "dispatch_agents.models",
        "InvocationStatus": "dispatch_agents.models",
        "KVGetResponse": "dispatch_agents.models",
        "KVListResponse": "dispatch_agents.models",
        "KVMemoryRecord": "dispatch_agents.models",
        "LLMFunctionCall": "dispatch_agents.models",
        "LLMToolCall": "dispatch_agents.models",
        "MemoryClient": "dispatch_agents.memory",
        "MemoryWriteResponse": "dispatch_agents.models",
        "SessionGetResponse": "dispatch_agents.models",
        "get_extra_llm_headers": "dispatch_agents.llm",
        "get_handler_metadata": "dispatch_agents.handlers",
        "get_handler_schemas": "dispatch_agents.handlers",
    },
    "dispatch_agents.config": {
        "DispatchConfig": "dispatch_agents.models",
        "DomainSelector": "dispatch_agents.models",
        "EgressConfig": "dispatch_agents.models",
        "MCPServerConfig": "dispatch_agents.models",
        "NetworkConfig": "dispatch_agents.models",
        "ResourceConfig": "dispatch_agents.models",
        "ResourceLimits": "dispatch_agents.models",
        "SecretConfig": "dispatch_agents.models",
        "VolumeConfig": "dispatch_agents.models",
        "VolumeMode": "dispatch_agents.models",
    },
    "dispatch_agents.events": {
        "BasePayload": "dispatch_agents.models",
        "HandlerMetadata": "dispatch_agents.models",
        "fn": "dispatch_agents.handlers",
        "get_handler_metadata": "dispatch_agents.handlers",
        "get_handler_schemas": "dispatch_agents.handlers",
        "init": "dispatch_agents.handlers",
        "invoke": "dispatch_agents.invocation",
        "on": "dispatch_agents.handlers",
    },
    "dispatch_agents.integrations.github.client": {
        "GitHubAppToken": "dispatch_agents.models",
    },
    "dispatch_agents.invocation": {
        "InvocationStatus": "dispatch_agents.models",
    },
    "dispatch_agents.llm": {
        "LLMFunctionCall": "dispatch_agents.models",
        "LLMMessage": "dispatch_agents.models",
        "LLMResponse": "dispatch_agents.models",
        "LLMToolCall": "dispatch_agents.models",
    },
    "dispatch_agents.mcp": {
        "McpHttpServerConfig": "dispatch_agents.models",
    },
    "dispatch_agents.memory": {
        "KVGetResponse": "dispatch_agents.models",
        "KVListResponse": "dispatch_agents.models",
        "MemoryWriteResponse": "dispatch_agents.models",
        "SessionGetResponse": "dispatch_agents.models",
    },
}


# Alias maps keyed by module name, populated by ``install_deprecated_public_aliases``.
# Held here rather than as a dynamic attribute on each patched module so the lookup
# stays fully typed (``ModuleType`` has no such attribute slot).
_INSTALLED_ALIASES: dict[str, AliasTargets] = {}


class _DeprecatedAliasModule(ModuleType):
    def __getattribute__(self, name: str) -> object:
        module_name = ModuleType.__getattribute__(self, "__name__")
        target_module = _INSTALLED_ALIASES.get(module_name, {}).get(name)
        if target_module is not None:
            return _resolve_deprecated_alias(module_name, name, target_module)
        return ModuleType.__getattribute__(self, name)


def install_deprecated_public_aliases() -> None:
    """Install all deprecated public import aliases."""
    for module_name, aliases in _all_aliases().items():
        module = import_module(module_name)
        _INSTALLED_ALIASES[module_name] = dict(aliases)
        module.__class__ = _DeprecatedAliasModule
        _extend_all(module, aliases)


def _all_aliases() -> dict[str, AliasTargets]:
    aliases = {module: dict(names) for module, names in _STATIC_ALIASES.items()}
    github_events = import_module("dispatch_agents.integrations.github.events")
    aliases["dispatch_agents.integrations.github"] = {
        name: "dispatch_agents.integrations.github.events"
        for name in github_events.__all__
        if name != "GitHubAppToken"
    }
    return aliases


def _resolve_deprecated_alias(
    module_name: str,
    name: str,
    target_module: str,
) -> Any:
    warnings.warn(
        f"{module_name}.{name} is deprecated; use {target_module}.{name} instead.",
        FutureWarning,
        stacklevel=3,
    )
    return getattr(import_module(target_module), name)


def _extend_all(module: ModuleType, aliases: AliasTargets) -> None:
    all_names = getattr(module, "__all__", None)
    if not isinstance(all_names, list):
        return
    for name in aliases:
        if name not in all_names:
            all_names.append(name)

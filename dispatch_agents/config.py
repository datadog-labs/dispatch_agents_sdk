"""Runtime configuration API for ``dispatch.yaml`` files.

Use the module-level :data:`config` object to read the current agent's runtime
configuration from ``dispatch.yaml`` and environment fallbacks.
"""

import functools as _functools
import os

import yaml as _yaml

from dispatch_agents._internal.config_validation import (
    extract_var_descriptions as _extract_var_descriptions,
)
from dispatch_agents._internal.config_validation import (
    unwrap_described_vars as _unwrap_described_vars,
)
from dispatch_agents.models import DispatchConfig as _DispatchConfig
from dispatch_agents.models import JsonObject as _JsonObject
from dispatch_agents.models import MCPServerConfig as _MCPServerConfig
from dispatch_agents.models import NetworkConfig as _NetworkConfig
from dispatch_agents.models import ResourceConfig as _ResourceConfig
from dispatch_agents.models import SecretConfig as _SecretConfig
from dispatch_agents.models import VolumeConfig as _VolumeConfig

__all__ = ["RuntimeConfig", "config"]

_DISPATCH_YAML_PATH = "/app/dispatch.yaml"


@_functools.lru_cache(maxsize=1)
def _load_runtime_config() -> _DispatchConfig:
    """Load and cache the current agent's runtime config."""
    path = os.environ.get("DISPATCH_CONFIG_PATH", _DISPATCH_YAML_PATH)
    raw: _JsonObject = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            loaded = _yaml.safe_load(f) or {}
            raw = loaded if isinstance(loaded, dict) else {}

    cfg = _DispatchConfig.model_validate(raw)

    updates: _JsonObject = {}
    if cfg.namespace is None:
        ns = os.environ.get("DISPATCH_NAMESPACE")
        if ns:
            updates["namespace"] = ns
    if cfg.agent_name is None:
        name = os.environ.get("DISPATCH_AGENT_NAME")
        if name:
            updates["agent_name"] = name
    if updates:
        cfg = cfg.model_copy(update=updates)

    return cfg


class RuntimeConfig:
    """Runtime view of the current agent's ``dispatch.yaml`` configuration.

    Configuration is loaded lazily and cached by the internal implementation.
    ``namespace`` and ``agent_name`` fall back to ``DISPATCH_NAMESPACE`` and
    ``DISPATCH_AGENT_NAME`` when a config file is not present, which keeps local
    dev runs usable outside a deployed container.

    Example::

        from dispatch_agents import config

        namespace = config.namespace
        agent_name = config.agent_name
        temperature = config.vars.get("temperature", 0.7)
        description = config.vars_descriptions.get("temperature")
    """

    @property
    def namespace(self) -> str | None:
        """Return the configured Dispatch namespace."""
        return _load_runtime_config().namespace

    @property
    def agent_name(self) -> str | None:
        """Return the configured agent name."""
        return _load_runtime_config().agent_name

    @property
    def entrypoint(self) -> str | None:
        """Return the configured Python entrypoint."""
        return _load_runtime_config().entrypoint

    @property
    def base_image(self) -> str | None:
        """Return the configured base container image."""
        return _load_runtime_config().base_image

    @property
    def system_packages(self) -> list[str] | None:
        """Return configured system packages."""
        return _load_runtime_config().system_packages

    @property
    def local_dependencies(self) -> dict[str, str] | None:
        """Return configured local path dependencies."""
        return _load_runtime_config().local_dependencies

    @property
    def env(self) -> dict[str, str] | None:
        """Return configured environment variables."""
        return _load_runtime_config().env

    @property
    def secrets(self) -> list[_SecretConfig] | None:
        """Return configured secret environment variables."""
        return _load_runtime_config().secrets

    @property
    def volumes(self) -> list[_VolumeConfig] | None:
        """Return configured persistent volumes."""
        return _load_runtime_config().volumes

    @property
    def mcp_servers(self) -> list[_MCPServerConfig] | None:
        """Return configured MCP server registry entries."""
        return _load_runtime_config().mcp_servers

    @property
    def resources(self) -> _ResourceConfig:
        """Return configured container resources."""
        return _load_runtime_config().resources

    @property
    def network(self) -> _NetworkConfig | None:
        """Return configured network policy settings."""
        return _load_runtime_config().network

    @property
    def llm_instrument(self) -> bool:
        """Return whether LLM calls route through the Dispatch sidecar proxy."""
        return _load_runtime_config().llm_instrument

    @property
    def log_level(self) -> str | None:
        """Return the configured SDK log level (DEBUG/INFO/WARNING/ERROR), or None."""
        return _load_runtime_config().log_level

    @property
    def vars(self) -> _JsonObject:
        """Return runtime variables from ``dispatch.yaml``.

        Described vars are unwrapped so ``{"value": 3, "description": "..."}``
        returns ``3`` in this mapping.
        """
        vars_data = _load_runtime_config().vars
        return _unwrap_described_vars(vars_data) if vars_data else {}

    @property
    def vars_descriptions(self) -> dict[str, str]:
        """Return descriptions for runtime variables that define them."""
        vars_data = _load_runtime_config().vars
        return _extract_var_descriptions(vars_data) if vars_data else {}


config = RuntimeConfig()
"""Runtime view of the current agent's ``dispatch.yaml`` configuration.

Use ``config.vars`` to read values from the ``vars`` section at runtime.
Described vars are unwrapped so ``{value: 3, description: "..."}`` returns ``3``.
"""

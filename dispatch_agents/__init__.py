"""Dispatch Agents Python SDK.

The root package re-exports the most common agent APIs for convenience:

* ``on``, ``fn``, and ``init`` from :mod:`dispatch_agents.handlers`
* ``emit_event`` from :mod:`dispatch_agents.events`
* ``invoke`` from :mod:`dispatch_agents.invocation`
* ``get_data_dir`` from :mod:`dispatch_agents.storage`
* ``get_mcp_client`` and ``get_mcp_servers_config`` from :mod:`dispatch_agents.mcp`
* ``config``, ``memory``, ``llm``, ``context``, and ``models`` as public modules

For canonical API ownership, browse the public submodules below.
Deprecated compatibility aliases from earlier SDK versions remain importable
from the root package, but new code should import them from their canonical
public submodules.
"""

from __future__ import annotations

from ._deprecated_public_api import install_deprecated_public_aliases
from ._internal.bootstrap import install_proto_import_path
from ._internal.dev import install_dev_mode_audit_hook

install_proto_import_path()

from . import context, llm, models
from .config import config
from .events import emit_event
from .handlers import fn, init, on
from .invocation import invoke
from .llm import extra_headers
from .mcp import get_mcp_client, get_mcp_servers_config
from .memory import memory
from .models import BasePayload, McpHttpServerConfig
from .storage import DisallowedWriteError, get_data_dir

__all__ = [
    "BasePayload",
    "config",
    "context",
    "DisallowedWriteError",
    "emit_event",
    "extra_headers",
    "fn",
    "get_data_dir",
    "get_mcp_client",
    "get_mcp_servers_config",
    "init",
    "invoke",
    "llm",
    "McpHttpServerConfig",
    "memory",
    "models",
    "on",
]

install_deprecated_public_aliases()
install_dev_mode_audit_hook()

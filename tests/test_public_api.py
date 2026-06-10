"""Public SDK import contract tests."""

import ast
import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


def test_root_public_api_contract():
    import dispatch_agents

    expected_core = {
        "BasePayload",
        "McpHttpServerConfig",
        "context",
        "config",
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
        "memory",
        "models",
        "on",
    }

    assert expected_core <= set(dispatch_agents.__all__)
    for name in expected_core:
        assert hasattr(dispatch_agents, name)


def test_get_data_dir_is_public_storage_api(monkeypatch, tmp_path):
    import dispatch_agents
    from dispatch_agents.storage import get_data_dir as storage_get_data_dir

    dev_data_dir = tmp_path / ".dispatch" / "dev-data"
    dev_data_dir.mkdir(parents=True)
    monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_data_dir))

    assert dispatch_agents.get_data_dir is storage_get_data_dir
    assert dispatch_agents.get_data_dir() == dev_data_dir / "data"


def test_dispatch_module_does_not_own_transport_helpers():
    import dispatch_agents._internal.dispatch as dispatch

    for name in (
        "_get_router_url",
        "_get_namespace",
        "_get_api_base_url",
        "_get_auth_headers",
    ):
        assert not hasattr(dispatch, name)


def test_root_does_not_expose_known_internals():
    import dispatch_agents

    internal_names = {
        "AgentServiceClient",
        "FeedbackSentiment",
        "FeedbackType",
        "FunctionMessage",
        "LongTermMemoryClient",
        "Message",
        "ShortTermMemoryClient",
        "TopicMessage",
        "dispatch_message",
        "get_current_invocation_id",
        "get_current_parent_id",
        "get_current_trace_id",
        "run_init_hook",
    }

    for name in internal_names:
        assert not hasattr(dispatch_agents, name)
        assert name not in dispatch_agents.__all__


def test_public_models_live_in_models_module():
    from dispatch_agents.models import (
        DispatchConfig,
        HandlerMetadata,
        InvocationContext,
        InvocationStatus,
        KVGetResponse,
        KVListResponse,
        KVMemoryRecord,
        LLMFunctionCall,
        LLMResponse,
        LLMToolCall,
        MemoryWriteResponse,
        ResourceLimits,
        SessionGetResponse,
    )

    assert DispatchConfig is not None
    assert HandlerMetadata is not None
    assert InvocationContext is not None
    assert InvocationStatus is not None
    assert LLMFunctionCall is not None
    assert LLMResponse is not None
    assert LLMToolCall is not None
    assert KVGetResponse is not None
    assert KVListResponse is not None
    assert KVMemoryRecord is not None
    assert MemoryWriteResponse is not None
    assert ResourceLimits is not None
    assert SessionGetResponse is not None


def test_behavior_modules_keep_canonical_runtime_exports():
    context = importlib.import_module("dispatch_agents.context")
    handlers = importlib.import_module("dispatch_agents.handlers")
    invocation = importlib.import_module("dispatch_agents.invocation")
    llm = importlib.import_module("dispatch_agents.llm")
    memory = importlib.import_module("dispatch_agents.memory")

    assert context.__all__ == ["current"]
    assert handlers.__all__ == [
        "fn",
        "get_handler_metadata",
        "get_handler_schemas",
        "init",
        "on",
    ]
    assert "invoke" in invocation.__all__
    assert not hasattr(context, "InvocationContext")
    assert not hasattr(handlers, "HandlerMetadata")
    assert llm.LLMClient
    assert memory.MemoryClient


def test_events_module_only_exposes_event_publication():
    import dispatch_agents.events as events

    assert "emit_event" in events.__all__
    assert hasattr(events, "emit_event")


def test_config_and_mcp_keep_canonical_runtime_exports():
    config = importlib.import_module("dispatch_agents.config")
    mcp = importlib.import_module("dispatch_agents.mcp")

    assert config.RuntimeConfig
    assert config.config
    assert mcp.MCPClient
    assert mcp.get_mcp_client
    assert mcp.get_mcp_servers_config


@pytest.mark.parametrize(
    ("module_name", "name", "target_module"),
    [
        ("dispatch_agents", "HandlerMetadata", "dispatch_agents.models"),
        ("dispatch_agents", "MemoryClient", "dispatch_agents.memory"),
        ("dispatch_agents", "get_handler_schemas", "dispatch_agents.handlers"),
        ("dispatch_agents.config", "DispatchConfig", "dispatch_agents.models"),
        ("dispatch_agents.events", "fn", "dispatch_agents.handlers"),
        ("dispatch_agents.events", "invoke", "dispatch_agents.invocation"),
        (
            "dispatch_agents.integrations.github",
            "PullRequestOpened",
            "dispatch_agents.integrations.github.events",
        ),
        ("dispatch_agents.invocation", "InvocationStatus", "dispatch_agents.models"),
        ("dispatch_agents.llm", "LLMResponse", "dispatch_agents.models"),
        ("dispatch_agents.mcp", "McpHttpServerConfig", "dispatch_agents.models"),
        ("dispatch_agents.memory", "KVGetResponse", "dispatch_agents.models"),
    ],
)
def test_deprecated_public_aliases_resolve_to_public_owners(
    module_name: str,
    name: str,
    target_module: str,
):
    module = importlib.import_module(module_name)
    target = getattr(importlib.import_module(target_module), name)

    assert name in module.__all__
    with pytest.warns(FutureWarning, match=f"{module_name}.{name} is deprecated"):
        assert getattr(module, name) is target


def test_all_deprecated_public_aliases_target_public_modules():
    from dispatch_agents._deprecated_public_api import _all_aliases

    for module_name, aliases in _all_aliases().items():
        module = importlib.import_module(module_name)
        for name, target_module in aliases.items():
            assert "._internal" not in target_module
            assert name in module.__all__
            target = getattr(importlib.import_module(target_module), name)
            with pytest.warns(
                FutureWarning, match=f"{module_name}.{name} is deprecated"
            ):
                assert getattr(module, name) is target


def test_deprecated_public_alias_warning_visible_from_imported_module(tmp_path: Path):
    """Deprecated aliases should warn during normal imported agent/module code."""

    imported_module = tmp_path / "uses_deprecated_alias.py"
    imported_module.write_text("from dispatch_agents.events import fn\n")
    runner = tmp_path / "runner.py"
    runner.write_text("import uses_deprecated_alias\n")

    result = subprocess.run(
        [sys.executable, str(runner)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "FutureWarning: dispatch_agents.events.fn is deprecated; "
        "use dispatch_agents.handlers.fn instead."
    ) in result.stderr


def _examples_dir() -> Path:
    """Locate the repo-root ``examples/`` directory relative to this test."""
    return Path(__file__).resolve().parents[2] / "examples"


def _imports_internal(source: str) -> bool:
    """Return True if ``source`` imports from ``dispatch_agents._internal``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "dispatch_agents._internal" or node.module.startswith(
                "dispatch_agents._internal."
            ):
                return True
            # ``from dispatch_agents import _internal [as x]`` reaches the private
            # package just as directly; the submodule name lives in ``names`` here.
            if node.module == "dispatch_agents" and any(
                alias.name == "_internal" for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dispatch_agents._internal" or alias.name.startswith(
                    "dispatch_agents._internal."
                ):
                    return True
    return False


def test_examples_do_not_import_internal():
    """External agent code (proxied by ``examples/``) must not reach into ``_internal``.

    First-party packages (``backend``, ``cli``) may, but example agents stand in for
    external users and are held to the public surface. See
    ``dispatch_agents/_internal/__init__.py`` for the policy.
    """
    examples = _examples_dir()
    if not examples.is_dir():
        pytest.skip("examples/ not present in this checkout")

    # Only the agents' own source counts. Skip installed dependencies (``.venv``),
    # CLI-generated build artifacts (``.dispatch``), and caches — those legitimately
    # contain ``_internal`` (the installed SDK itself) and are not authored code.
    skip_dirs = {
        ".venv",
        ".dispatch",
        "__pycache__",
        "build",
        "dist",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
    }

    offenders = [
        str(path.relative_to(examples))
        for path in examples.rglob("*.py")
        if not (skip_dirs & set(path.relative_to(examples).parts))
        and _imports_internal(path.read_text())
    ]
    assert not offenders, (
        "example agents must use the public surface, not dispatch_agents._internal: "
        f"{offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "from dispatch_agents._internal import dispatch",
        "from dispatch_agents._internal.dispatch import invoke",
        "import dispatch_agents._internal",
        "import dispatch_agents._internal.dispatch",
        "import dispatch_agents._internal as guts",
        # The submodule-import form that previously slipped past the guard.
        "from dispatch_agents import _internal",
        "from dispatch_agents import _internal as guts",
    ],
)
def test_imports_internal_detects_violations(source: str):
    assert _imports_internal(source) is True


@pytest.mark.parametrize(
    "source",
    [
        "from dispatch_agents import fn, on, BasePayload",
        "from dispatch_agents.models import InvocationResult",
        "import dispatch_agents",
        # A package merely named similarly must not trip the guard.
        "import dispatch_agents._internalish",
    ],
)
def test_imports_internal_allows_public_surface(source: str):
    assert _imports_internal(source) is False


# Documented canonical public names a normal agent reaches. These must stay
# canonical and never resolve through the deprecation shim. Intentionally a
# curated subset (not every ``__all__`` entry); the assertions below use
# subset/disjoint semantics so adding new public exports never breaks this.
_CANONICAL_PUBLIC_SURFACE: dict[str, set[str]] = {
    "dispatch_agents": {
        "BasePayload",
        "DisallowedWriteError",
        "McpHttpServerConfig",
        "config",
        "context",
        "emit_event",
        "extra_headers",
        "fn",
        "get_data_dir",
        "get_mcp_client",
        "get_mcp_servers_config",
        "init",
        "invoke",
        "llm",
        "memory",
        "models",
        "on",
    },
    "dispatch_agents.handlers": {
        "fn",
        "get_handler_metadata",
        "get_handler_schemas",
        "init",
        "on",
    },
    "dispatch_agents.events": {"emit_event"},
    "dispatch_agents.invocation": {"invoke"},
    "dispatch_agents.context": {"current"},
    "dispatch_agents.llm": {
        "LLMClient",
        "chat",
        "extra_headers",
        "inference",
        "llm",
        "parse_json",
    },
    "dispatch_agents.memory": {
        "LongTermMemoryClient",
        "MemoryClient",
        "ShortTermMemoryClient",
        "memory",
    },
    "dispatch_agents.config": {"RuntimeConfig", "config"},
    "dispatch_agents.mcp": {
        "MCPClient",
        "get_mcp_client",
        "get_mcp_servers_config",
    },
}


def test_canonical_public_api_is_never_a_deprecation_alias():
    """The documented public surface must not resolve through the deprecation shim.

    Without this guard, an alias mistakenly added to a *canonical* re-export path
    would emit a ``FutureWarning`` to every user on an ordinary import while the
    rest of the suite stayed green.
    For each documented name we assert it is (a) absent from the alias maps and
    (b) present and accessible without tripping a user-visible warning.
    """
    from dispatch_agents._deprecated_public_api import _all_aliases

    aliases = _all_aliases()
    for module_name, names in _CANONICAL_PUBLIC_SURFACE.items():
        module = importlib.import_module(module_name)
        alias_names = set(aliases.get(module_name, {}))

        collisions = names & alias_names
        assert not collisions, (
            f"{module_name} lists {sorted(collisions)} as both a canonical export "
            "and a deprecation alias; canonical public names must never be deprecated."
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            for name in sorted(names):
                assert getattr(module, name) is not None

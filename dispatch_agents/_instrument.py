"""Auto-instrumentation for LLM SDK calls.

Patches httpx and requests to inject Dispatch trace context headers on
requests destined for the sidecar proxy. This enables automatic trace
correlation for any LLM SDK (OpenAI, Anthropic, etc.) without user
code changes.

Usage:
    Called automatically by grpc_listener.py before user code imports.
    Not intended to be called directly by user code.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PROXY_HOST = ""  # Set at instrument time from env


def _is_proxy_bound(url: Any) -> bool:
    """Check if a request URL targets the sidecar proxy."""
    if not PROXY_HOST:
        return False
    return str(url).startswith(PROXY_HOST)


def _get_context_headers() -> dict[str, str]:
    """Build trace context headers from current execution context.

    Reads from contextvars set by dispatch_agents.events during handler
    execution, so headers are automatically scoped to the current invocation.

    Also serializes any extra LLM headers (set via extra_headers() context
    manager) into a single ``X-Dispatch-Extra-Headers`` JSON header so
    they can be forwarded by the sidecar proxy without polluting the
    header namespace.
    """
    import json

    from .events import get_current_invocation_id, get_current_trace_id
    from .llm import get_extra_llm_headers

    headers: dict[str, str] = {}

    trace_id = get_current_trace_id()
    if trace_id:
        headers["X-Dispatch-Trace-Id"] = trace_id

    invocation_id = get_current_invocation_id()
    if invocation_id:
        headers["X-Dispatch-Invocation-Id"] = invocation_id

    agent_name = os.environ.get("DISPATCH_AGENT_NAME", "")
    if agent_name:
        headers["X-Dispatch-Agent-Name"] = agent_name

    extra = get_extra_llm_headers()
    if extra:
        headers["X-Dispatch-Extra-Headers"] = json.dumps(extra)

    return headers


def auto_instrument() -> None:
    """Patch httpx, requests, and subprocess to inject trace context.

    - httpx/requests: Injects headers on proxy-bound requests (in-process SDKs)
    - subprocess: Injects ANTHROPIC_CUSTOM_HEADERS env var so child processes
      (e.g. Claude Agent SDK CLI) include trace context in their HTTP requests

    Safe to call multiple times — patches are idempotent.
    """
    global PROXY_HOST
    PROXY_HOST = os.environ.get("DISPATCH_LLM_PROXY_URL", "")

    if not PROXY_HOST:
        logger.debug("DISPATCH_LLM_PROXY_URL not set, skipping instrumentation")
        return

    _patch_httpx()
    _patch_requests()
    _patch_subprocess()

    logger.info("Auto-instrumentation enabled for proxy at %s", PROXY_HOST)


def _patch_httpx() -> None:
    """Patch httpx.Client.send and httpx.AsyncClient.send."""
    try:
        import httpx
    except ImportError:
        return

    # Patch sync client
    if not getattr(httpx.Client.send, "_dispatch_patched", False):
        _original_sync_send = httpx.Client.send

        def _patched_sync_send(self: Any, request: Any, **kwargs: Any) -> Any:
            if _is_proxy_bound(request.url):
                for key, value in _get_context_headers().items():
                    request.headers[key] = value
            return _original_sync_send(self, request, **kwargs)

        _patched_sync_send._dispatch_patched = True  # type: ignore[attr-defined]
        httpx.Client.send = _patched_sync_send  # type: ignore[method-assign]

    # Patch async client
    if not getattr(httpx.AsyncClient.send, "_dispatch_patched", False):
        _original_async_send = httpx.AsyncClient.send

        async def _patched_async_send(self: Any, request: Any, **kwargs: Any) -> Any:
            if _is_proxy_bound(request.url):
                for key, value in _get_context_headers().items():
                    request.headers[key] = value
            return await _original_async_send(self, request, **kwargs)

        _patched_async_send._dispatch_patched = True  # type: ignore[attr-defined]
        httpx.AsyncClient.send = _patched_async_send  # type: ignore[method-assign]


def _patch_requests() -> None:
    """Patch requests.Session.send for libraries using requests (e.g. Google SDK)."""
    try:
        import requests
    except ImportError:
        return

    if not getattr(requests.Session.send, "_dispatch_patched", False):
        _original_send = requests.Session.send

        def _patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
            if _is_proxy_bound(request.url):
                for key, value in _get_context_headers().items():
                    request.headers[key] = value
            return _original_send(self, request, **kwargs)

        _patched_send._dispatch_patched = True  # type: ignore[attr-defined]
        requests.Session.send = _patched_send  # type: ignore[method-assign]


def _build_trace_custom_headers() -> str | None:
    """Build ANTHROPIC_CUSTOM_HEADERS value from current trace context.

    Returns a newline-separated header string, or None if no trace context.
    The Claude CLI reads this env var and includes the headers on every
    HTTP request it makes to ANTHROPIC_BASE_URL (our sidecar proxy).
    """
    from .events import get_current_invocation_id, get_current_trace_id

    parts: list[str] = []
    trace_id = get_current_trace_id()
    if trace_id:
        parts.append(f"X-Dispatch-Trace-Id: {trace_id}")
    invocation_id = get_current_invocation_id()
    if invocation_id:
        parts.append(f"X-Dispatch-Invocation-Id: {invocation_id}")
    return "\n".join(parts) if parts else None


def _inject_trace_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """Inject trace context headers into a subprocess env dict.

    Sets provider-specific custom header env vars so CLI tools (Claude CLI,
    Gemini CLI, etc.) include trace context in their HTTP requests.
    Each subprocess gets its own env copy at fork time.

    If env is None (inherit parent env), creates a copy of os.environ.
    Concurrent-safe: reads from ContextVars which are per-async-task.

    Provider support:
      - ANTHROPIC_CUSTOM_HEADERS: Claude CLI (newline-separated headers)
      - GEMINI_CLI_CUSTOM_HEADERS: Gemini CLI (same format)
      - OpenAI/Cohere/Mistral: No CLI custom header env var — in-process
        SDKs are covered by httpx/requests patches instead.
    """
    custom_headers = _build_trace_custom_headers()
    if not custom_headers:
        return env

    import uuid

    if env is None:
        env = os.environ.copy()
    else:
        env = dict(env)  # Don't mutate the caller's dict

    # Add a unique subprocess ID so the backend can group LLM calls
    # by subprocess within a trace (e.g. multiple subagents in one invocation)
    subprocess_id = str(uuid.uuid4())
    custom_headers += f"\nX-Dispatch-Subprocess-Id: {subprocess_id}"

    # Provider CLIs that support custom headers via env var
    env["ANTHROPIC_CUSTOM_HEADERS"] = custom_headers
    env["GEMINI_CLI_CUSTOM_HEADERS"] = custom_headers
    return env


def _patch_subprocess() -> None:
    """Patch subprocess.Popen to inject trace context into child process env.

    This ensures subprocesses (e.g. Claude Agent SDK CLI) automatically
    include trace headers in their HTTP requests. ContextVars are read
    at spawn time, so concurrent invocations each get the correct trace_id.
    """
    import subprocess

    if not getattr(subprocess.Popen.__init__, "_dispatch_patched", False):
        _original_init = subprocess.Popen.__init__

        def _patched_init(self: Any, args: Any, **kwargs: Any) -> None:
            kwargs["env"] = _inject_trace_env(kwargs.get("env"))
            return _original_init(self, args, **kwargs)

        _patched_init._dispatch_patched = True  # type: ignore[attr-defined]
        subprocess.Popen.__init__ = _patched_init  # type: ignore[assignment,method-assign]

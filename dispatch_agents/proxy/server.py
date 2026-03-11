"""LLM Sidecar Proxy — thin pass-through to the Dispatch backend.

A minimal Starlette server on localhost that intercepts OpenAI/Anthropic
SDK calls and forwards them **unmodified** to the backend's /llm/proxy endpoint.
The backend handles all format conversion, credentials, LLM routing,
cost tracking, and telemetry.

When the backend has no LLM provider configured, the proxy falls back to
calling the provider directly using the developer's original API key (saved
in _DISPATCH_ORIGINAL_* env vars by grpc_listener.py). Successful fallback
calls are logged to /llm/log for observability.

Unsupported endpoints (embeddings, models list, audio, images, etc.) are
forwarded via the backend's /llm/passthrough endpoint for credential injection
without full observability parsing.

The proxy only adds metadata (agent_name, trace context) — no message
format conversion happens here.

Started as a subprocess by grpc_listener.py before user code loads.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpcore
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from dispatch_agents.proxy.sse_utils import (
    StreamingUsageCollector,
)

logger = logging.getLogger(__name__)

# Cache: trace_id → True means "backend has no LLM config, use fallback".
# Within one invocation (same trace_id), only the first call checks the backend.
# New invocations (new trace_id) get a fresh check.
_fallback_traces: set[str] = set()

# Provider base URL and auth header configuration.
# "base_url" is the provider's root — specific endpoint paths are appended at call time.
_PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com",
        "key_env": "_DISPATCH_ORIGINAL_OPENAI_API_KEY",
        "key_name": "OPENAI_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "key_env": "_DISPATCH_ORIGINAL_ANTHROPIC_API_KEY",
        "key_name": "ANTHROPIC_API_KEY",
        "auth_header": "x-api-key",
        "auth_prefix": "",
    },
}

# Fixed endpoint paths for the supported (full-observability) proxy routes.
_PROVIDER_ENDPOINT: dict[str, str] = {
    "openai": "/v1/chat/completions",
    "anthropic": "/v1/messages",
}

# Sidecar route paths (provider-prefixed).
# Used by grpc_listener.py to set OPENAI_BASE_URL / ANTHROPIC_BASE_URL,
# and by tests to hit the correct routes.
OPENAI_CHAT_ROUTE = "/openai/v1/chat/completions"
OPENAI_RESPONSES_ROUTE = "/openai/v1/responses"
ANTHROPIC_MESSAGES_ROUTE = "/anthropic/v1/messages"


def _get_backend_proxy_url() -> str:
    """Get the backend LLM proxy URL."""
    router_url = os.environ.get("BACKEND_URL", "http://dispatch.api:8000")
    namespace = os.environ.get("DISPATCH_NAMESPACE", "dev")
    return f"{router_url}/api/unstable/namespace/{namespace}/llm/proxy"


def _get_backend_passthrough_url() -> str:
    """Get the backend LLM passthrough URL."""
    router_url = os.environ.get("BACKEND_URL", "http://dispatch.api:8000")
    namespace = os.environ.get("DISPATCH_NAMESPACE", "dev")
    return f"{router_url}/api/unstable/namespace/{namespace}/llm/passthrough"


def _get_backend_log_url() -> str:
    """Get the backend LLM log URL for observability."""
    router_url = os.environ.get("BACKEND_URL", "http://dispatch.api:8000")
    namespace = os.environ.get("DISPATCH_NAMESPACE", "dev")
    return f"{router_url}/api/unstable/namespace/{namespace}/llm/log"


def _get_auth_headers() -> dict[str, str]:
    """Build auth headers for backend requests."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("DISPATCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _is_not_configured_error(status_code: int, body: bytes) -> bool:
    """Check if a backend response indicates no LLM provider is configured.

    Only matches the specific "not configured" 400 error from the backend,
    not other 400 errors (budget exceeded, invalid request, etc.).

    The backend /llm/proxy endpoint returns errors in SDK format:
      OpenAI:    {"error": {"message": "No LLM providers configured..."}}
      Anthropic: {"type": "error", "error": {"message": "No LLM providers configured..."}}
    Other backend endpoints use: {"detail": "..."}
    """
    if status_code != 400:
        return False
    try:
        data = json.loads(body)
        # Extract error message — check SDK format first, then FastAPI format
        message = ""
        if isinstance(data.get("error"), dict):
            message = data["error"].get("message", "")
        elif isinstance(data.get("detail"), str):
            message = data["detail"]
        if not message:
            return False
        lower = message.lower()
        return "no llm providers configured" in lower or (
            "provider" in lower and "not configured" in lower
        )
    except (json.JSONDecodeError, AttributeError):
        logger.debug("Failed to parse error body in _is_not_configured_error")
    return False


def _is_auth_error(status_code: int) -> bool:
    """Check if a backend response indicates the LLM provider rejected the API key.

    The sidecar authenticates to the backend with DISPATCH_API_KEY, so a
    401/403 from the proxy endpoint is almost always a provider auth error
    forwarded by the backend (not Dispatch's own auth rejecting us).
    """
    return status_code in (401, 403)


def _log_backend_error(status_code: int, body: bytes, provider_format: str) -> None:
    """Log backend proxy errors clearly so developers can debug agent failures.

    Extracts the error message from the backend's JSON response and logs it
    at WARNING level. This surfaces errors like "provider not configured",
    "budget exceeded", "invalid model", etc. in the agent container logs
    instead of burying them inside SDK exception wrappers.
    """
    message = "(could not parse error body)"
    try:
        data = json.loads(body)
        if isinstance(data.get("error"), dict):
            message = data["error"].get("message", str(data["error"]))
        elif isinstance(data.get("detail"), str):
            message = data["detail"]
        elif isinstance(data.get("error"), str):
            message = data["error"]
        else:
            message = body.decode("utf-8", errors="replace")[:500]
    except (json.JSONDecodeError, UnicodeDecodeError):
        message = body[:500].decode("utf-8", errors="replace")

    logger.warning(
        "LLM proxy error from backend (HTTP %d, %s): %s",
        status_code,
        provider_format,
        message,
    )


# Headers that are internal to the sidecar proxy and should NOT be forwarded
# to the provider.  Everything else the SDK sends gets passed through so that
# new provider headers (anthropic-beta, openai-beta, etc.) work automatically
# without sidecar changes.
_STRIP_HEADERS = {
    "host",
    "content-length",
    "content-type",
    "transfer-encoding",
    "connection",
    "accept",
    "accept-encoding",
    "user-agent",
    "authorization",
    "x-api-key",
}
_STRIP_PREFIXES = ("x-dispatch-",)


def _extract_trace_context(
    request: Request,
) -> tuple[dict[str, str], str | None, str | None, str | None, dict[str, str] | None]:
    """Extract trace context and extra headers from the incoming request.

    Returns (headers_dict, trace_id, invocation_id, subprocess_id, extra_headers).
    """
    headers = dict(request.headers)
    trace_id = headers.get("x-dispatch-trace-id")
    invocation_id = headers.get("x-dispatch-invocation-id")
    subprocess_id = headers.get("x-dispatch-subprocess-id")

    extra_headers_json = headers.get("x-dispatch-extra-headers")
    extra_headers: dict[str, str] | None = None
    if extra_headers_json:
        try:
            extra_headers = json.loads(extra_headers_json)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Failed to parse X-Dispatch-Extra-Headers, ignoring")

    # Forward all SDK headers except internal/transport ones so that provider
    # headers (anthropic-version, anthropic-beta, openai-beta, etc.) pass
    # through automatically without needing an explicit allowlist.
    for name, value in headers.items():
        if name in _STRIP_HEADERS:
            continue
        if any(name.startswith(p) for p in _STRIP_PREFIXES):
            continue
        if extra_headers is None:
            extra_headers = {}
        extra_headers.setdefault(name, value)

    return headers, trace_id, invocation_id, subprocess_id, extra_headers


async def _call_provider_directly(
    body: dict[str, Any],
    provider_format: str,
    extra_headers: dict[str, str] | None = None,
    endpoint: str | None = None,
) -> Response:
    """Call the LLM provider directly using saved original API keys.

    Used for supported endpoints (chat/messages) when backend has no config.
    Returns a 401 with actionable instructions if no original key exists.
    """
    config = _PROVIDER_CONFIG.get(provider_format)
    if not config:
        return JSONResponse(
            {"error": {"message": f"Unsupported provider: {provider_format}"}},
            status_code=500,
        )

    api_key = os.environ.get(config["key_env"])
    if not api_key:
        # Scenario C: no provider configured AND no developer key
        msg = (
            f"No LLM provider configured in Dispatch and no {config['key_name']} found. "
            f"Either:\n"
            f"  1. Configure a provider: dispatch llm setup\n"
            f"  2. Add your own key in dispatch.yaml secrets:\n"
            f"     secrets:\n"
            f"       - name: {config['key_name']}\n"
            f"         secret_id: /your/secret/path"
        )
        if provider_format == "anthropic":
            return JSONResponse(
                {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": msg},
                },
                status_code=401,
            )
        else:
            return JSONResponse(
                {"error": {"message": msg, "type": "auth_error", "code": "no_api_key"}},
                status_code=401,
            )

    # Build provider-specific headers (defaults first, then extra_headers override)
    # Provider-specific headers (anthropic-version, anthropic-beta, etc.)
    # come from extra_headers, captured from the SDK's original request.
    headers: dict[str, str] = {"Content-Type": "application/json"}
    headers[config["auth_header"]] = f"{config['auth_prefix']}{api_key}"
    if extra_headers:
        headers.update(extra_headers)

    url = config["base_url"] + (endpoint or _PROVIDER_ENDPOINT.get(provider_format, ""))

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )
    except (httpx.RequestError, httpcore.ConnectError, OSError):
        if provider_format == "anthropic":
            return JSONResponse(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Cannot reach {url}"},
                },
                status_code=502,
            )
        else:
            return JSONResponse(
                {
                    "error": {
                        "message": f"Cannot reach {url}",
                        "type": "connection_error",
                    }
                },
                status_code=502,
            )


async def _call_provider_passthrough(
    provider_format: str,
    path: str,
    method: str,
    body: bytes | None,
    query_string: str,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    """Call the provider directly for passthrough (unsupported) endpoints.

    Unlike _call_provider_directly(), this handles arbitrary paths and methods
    (GET, POST, DELETE) and forwards raw bytes without JSON parsing.
    """
    config = _PROVIDER_CONFIG.get(provider_format)
    if not config:
        return JSONResponse(
            {"error": {"message": f"Unsupported provider: {provider_format}"}},
            status_code=500,
        )

    api_key = os.environ.get(config["key_env"])
    if not api_key:
        msg = (
            f"No LLM provider configured in Dispatch and no {config['key_name']} found. "
            f"Set DISPATCH_LLM_INSTRUMENT=false or configure a provider."
        )
        return JSONResponse({"error": {"message": msg}}, status_code=401)

    # Build URL
    url = config["base_url"] + path
    if query_string:
        url += f"?{query_string}"

    # Build headers (defaults first, then extra_headers override)
    headers: dict[str, str] = {}
    headers[config["auth_header"]] = f"{config['auth_prefix']}{api_key}"
    if body:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                method=method,
                url=url,
                content=body,
                headers=headers,
            )
        # Preserve the content type from the provider response
        content_type = resp.headers.get("content-type", "application/json")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=content_type,
        )
    except (httpx.RequestError, httpcore.ConnectError, OSError):
        return JSONResponse(
            {"error": {"message": f"Cannot reach provider at {url}"}},
            status_code=502,
        )


async def _log_fallback_call(
    body: dict[str, Any],
    provider_format: str,
    response_body: bytes,
    trace_id: str | None,
    invocation_id: str | None,
    latency_ms: int,
) -> None:
    """Fire-and-forget: log a fallback LLM call to the backend for observability.

    Best-effort — never fails the user's request.
    """
    try:
        resp_data = json.loads(response_body)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse response body in _log_fallback_call, skipping")
        return

    payload: dict[str, Any] = {
        "input_messages": body.get("messages", []),
        "provider": provider_format,
        "latency_ms": latency_ms,
    }

    # Extract model and tokens based on provider format
    if provider_format == "openai":
        payload["model"] = resp_data.get("model", body.get("model", "unknown"))
        usage = resp_data.get("usage", {})
        payload["input_tokens"] = usage.get("prompt_tokens", 0)
        payload["output_tokens"] = usage.get("completion_tokens", 0)
        choices = resp_data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            payload["response_content"] = msg.get("content")
            payload["finish_reason"] = choices[0].get("finish_reason", "stop")
    elif provider_format == "anthropic":
        payload["model"] = resp_data.get("model", body.get("model", "unknown"))
        usage = resp_data.get("usage", {})
        payload["input_tokens"] = usage.get("input_tokens", 0)
        payload["output_tokens"] = usage.get("output_tokens", 0)
        # Extract text content from Anthropic response
        for block in resp_data.get("content", []):
            if block.get("type") == "text":
                payload["response_content"] = block.get("text")
                break
        stop_reason = resp_data.get("stop_reason", "end_turn")
        finish_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        payload["finish_reason"] = finish_map.get(stop_reason, stop_reason)

    if trace_id:
        payload["trace_id"] = trace_id
    if invocation_id:
        payload["invocation_id"] = invocation_id

    agent_name = os.environ.get("DISPATCH_AGENT_NAME")
    if agent_name:
        payload["agent_name"] = agent_name

    try:
        log_url = _get_backend_log_url()
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(log_url, json=payload, headers=_get_auth_headers())
    except (httpx.RequestError, httpcore.ConnectError, OSError):
        logger.debug("Failed to log fallback LLM call (best-effort)", exc_info=True)


async def health(request: Request) -> JSONResponse:
    """Health check for readiness probes."""
    return JSONResponse({"status": "ok"})


# =============================================================================
# Supported endpoints — full observability (proxy → backend /llm/proxy)
# =============================================================================


async def _proxy_to_backend(
    request: Request, provider_format: str, endpoint: str
) -> Response:
    """Forward an SDK request to the backend, adding metadata.

    The backend handles all format conversion, credential injection,
    and response formatting. This proxy just adds trace context and
    agent identity.

    If the backend returns a "provider not configured" error, falls back
    to calling the provider directly using the developer's original API key.
    """
    body = await request.json()
    _, trace_id, invocation_id, subprocess_id, extra_headers = _extract_trace_context(
        request
    )
    is_streaming = body.get("stream", False)

    # Check fallback cache — skip backend if we already know it's not configured
    if trace_id and trace_id in _fallback_traces:
        if is_streaming:
            return await _call_provider_directly_streaming(
                body,
                provider_format,
                extra_headers,
                endpoint,
                trace_id,
                invocation_id,
            )
        start = time.monotonic()
        fallback_resp = await _call_provider_directly(
            body, provider_format, extra_headers, endpoint
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if fallback_resp.status_code < 400:
            asyncio.create_task(
                _log_fallback_call(
                    body,
                    provider_format,
                    fallback_resp.body,
                    trace_id,
                    invocation_id,
                    elapsed_ms,
                )
            )
        return fallback_resp

    # Build the backend proxy request — wrap raw SDK body with metadata
    backend_payload: dict[str, Any] = {
        "provider_format": provider_format,
        "body": body,
        "endpoint": endpoint,
    }

    if trace_id:
        backend_payload["trace_id"] = trace_id
    if invocation_id:
        backend_payload["invocation_id"] = invocation_id
    if subprocess_id:
        backend_payload["subprocess_id"] = subprocess_id
    if extra_headers:
        backend_payload["extra_headers"] = extra_headers

    # Inject agent identity
    agent_name = os.environ.get("DISPATCH_AGENT_NAME")
    if agent_name:
        backend_payload["agent_name"] = agent_name

    # Streaming path — pipe SSE from backend to SDK
    if is_streaming:
        return await _proxy_to_backend_streaming(
            backend_payload,
            provider_format,
            trace_id,
        )

    # Non-streaming path — forward to backend and return response
    backend_url = _get_backend_proxy_url()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                backend_url,
                json=backend_payload,
                headers=_get_auth_headers(),
            )
    except (httpx.RequestError, httpcore.ConnectError, OSError):
        # Return error in the format the SDK expects
        if provider_format == "anthropic":
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Cannot reach Dispatch backend. Is the router running?",
                    },
                },
                status_code=502,
            )
        else:
            return JSONResponse(
                {
                    "error": {
                        "message": "Cannot reach Dispatch backend. Is the router running?",
                        "type": "connection_error",
                        "code": "backend_unavailable",
                    }
                },
                status_code=502,
            )

    # Detect "not configured" and fall back to direct provider call
    if _is_not_configured_error(resp.status_code, resp.content):
        logger.info(
            "Backend has no LLM provider configured, falling back to direct %s call",
            provider_format,
        )
        if trace_id:
            _fallback_traces.add(trace_id)
        start = time.monotonic()
        fallback_resp = await _call_provider_directly(
            body, provider_format, extra_headers, endpoint
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Log for observability (fire-and-forget, only on success)
        if fallback_resp.status_code < 400:
            asyncio.create_task(
                _log_fallback_call(
                    body,
                    provider_format,
                    fallback_resp.body,
                    trace_id,
                    invocation_id,
                    elapsed_ms,
                )
            )
        return fallback_resp

    # Detect provider auth error — fall back to agent's own key if available
    if _is_auth_error(resp.status_code):
        config = _PROVIDER_CONFIG.get(provider_format)
        has_own_key = config and os.environ.get(config["key_env"])
        if has_own_key:
            logger.warning(
                "Provider auth failed (HTTP %d), falling back to agent's own %s key",
                resp.status_code,
                provider_format,
            )
            if trace_id:
                _fallback_traces.add(trace_id)
            start = time.monotonic()
            fallback_resp = await _call_provider_directly(
                body, provider_format, extra_headers, endpoint
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if fallback_resp.status_code < 400:
                asyncio.create_task(
                    _log_fallback_call(
                        body,
                        provider_format,
                        fallback_resp.body,
                        trace_id,
                        invocation_id,
                        elapsed_ms,
                    )
                )
            return fallback_resp
        logger.warning(
            "Provider auth failed (HTTP %d) and no fallback %s key available",
            resp.status_code,
            provider_format,
        )

    # Surface backend errors clearly in agent logs so developers can debug
    if resp.status_code >= 400:
        _log_backend_error(resp.status_code, resp.content, provider_format)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
    )


# =============================================================================
# Streaming helpers
# =============================================================================


async def _proxy_to_backend_streaming(
    backend_payload: dict[str, Any],
    provider_format: str,
    trace_id: str | None,
) -> Response:
    """Stream SSE from the backend to the SDK caller.

    The backend's /llm/proxy now returns text/event-stream when stream=true.
    We pipe the bytes through without parsing — the backend handles cost tracking.
    Falls back to direct provider call if backend is unreachable or not configured.
    """

    backend_url = _get_backend_proxy_url()
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def stream_from_backend():
        try:
            async with client.stream(
                "POST",
                backend_url,
                json=backend_payload,
                headers=_get_auth_headers(),
            ) as resp:
                # If the backend returns an error, it's JSON (not SSE)
                if resp.status_code >= 400:
                    error_body = b""
                    async for chunk in resp.aiter_bytes():
                        error_body += chunk

                    # Check if it's a "not configured" error — fall back
                    if _is_not_configured_error(resp.status_code, error_body):
                        logger.info(
                            "Backend has no LLM config, falling back to direct %s streaming",
                            provider_format,
                        )
                        if trace_id:
                            _fallback_traces.add(trace_id)
                    elif _is_auth_error(resp.status_code):
                        # Provider key rejected — cache trace so next call
                        # goes direct via _call_provider_directly_streaming
                        config = _PROVIDER_CONFIG.get(provider_format)
                        has_fallback = config and os.environ.get(config["key_env"])
                        if has_fallback and trace_id:
                            _fallback_traces.add(trace_id)
                        logger.warning(
                            "Provider auth failed (HTTP %d)%s",
                            resp.status_code,
                            ", will fall back on next call"
                            if has_fallback
                            else f", no fallback {provider_format} key available",
                        )
                        _log_backend_error(
                            resp.status_code, error_body, provider_format
                        )
                    else:
                        # Surface non-config backend errors in agent logs
                        _log_backend_error(
                            resp.status_code, error_body, provider_format
                        )
                    yield error_body
                    return

                # Stream SSE bytes through
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except (httpx.RequestError, httpcore.ConnectError, OSError) as e:
            logger.warning("Backend unreachable for streaming, error: %s", e)
            error_json = json.dumps(
                _format_sse_error(provider_format, "Cannot reach backend")
            )
            yield error_json.encode()
        finally:
            await client.aclose()

    return StreamingResponse(
        stream_from_backend(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _format_sse_error(provider_format: str, message: str) -> dict[str, Any]:
    """Format an error in the provider's expected format."""
    if provider_format == "anthropic":
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
    return {
        "error": {"message": message, "type": "backend_error", "code": "502"},
    }


async def _call_provider_directly_streaming(
    body: dict[str, Any],
    provider_format: str,
    extra_headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> Response:
    """Call the LLM provider directly with streaming (fallback when backend has no config).

    Uses raw httpx streaming (not litellm) since litellm is not available in the SDK.
    Buffers SSE events for usage extraction and logs to /llm/log after stream completes.
    """

    config = _PROVIDER_CONFIG.get(provider_format)
    if not config:
        return JSONResponse(
            {"error": {"message": f"Unsupported provider: {provider_format}"}},
            status_code=500,
        )

    api_key = os.environ.get(config["key_env"])
    if not api_key:
        msg = (
            f"No LLM provider configured in Dispatch and no {config['key_name']} found. "
            f"Either:\n"
            f"  1. Configure a provider: dispatch llm setup\n"
            f"  2. Add your own key in dispatch.yaml secrets"
        )
        if provider_format == "anthropic":
            return JSONResponse(
                {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": msg},
                },
                status_code=401,
            )
        return JSONResponse(
            {"error": {"message": msg, "type": "auth_error", "code": "no_api_key"}},
            status_code=401,
        )

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if provider_format == "anthropic":
        headers["x-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    url = config["base_url"] + (endpoint or _PROVIDER_ENDPOINT.get(provider_format, ""))
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    buffered_lines: list[str] = []
    start_time = time.monotonic()

    async def stream_and_log():
        try:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                    return

                async for line in resp.aiter_lines():
                    sse_line = line + "\n"
                    yield sse_line.encode()
                    buffered_lines.append(line)
        except (httpx.RequestError, httpcore.ConnectError, OSError) as exc:
            logger.warning("Direct provider streaming failed: %s", exc)
        finally:
            await client.aclose()
            # Log to backend for observability (fire-and-forget)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            asyncio.create_task(
                _log_fallback_streaming_call(
                    body,
                    provider_format,
                    buffered_lines,
                    trace_id,
                    invocation_id,
                    elapsed_ms,
                )
            )

    return StreamingResponse(
        stream_and_log(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _log_fallback_streaming_call(
    body: dict[str, Any],
    provider_format: str,
    buffered_lines: list[str],
    trace_id: str | None,
    invocation_id: str | None,
    latency_ms: int,
) -> None:
    """Fire-and-forget: extract usage from buffered SSE lines and log to backend."""

    try:
        # Parse the buffered SSE lines into events
        collector = StreamingUsageCollector(provider_format)
        current_event_type: str | None = None
        for line in buffered_lines:
            stripped = line.strip()
            if stripped.startswith("event:"):
                current_event_type = stripped[6:].strip()
            elif stripped.startswith("data:"):
                data_str = stripped[5:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                    obj["_event_type"] = current_event_type
                    collector.observe(obj)
                except (json.JSONDecodeError, ValueError):
                    logger.debug(
                        "Failed to parse SSE data for usage: %s", data_str[:100]
                    )
        collector.finalize()

        payload: dict[str, Any] = {
            "input_messages": body.get("messages", []),
            "provider": provider_format,
            "latency_ms": latency_ms,
            "model": collector.model or body.get("model", "unknown"),
            "input_tokens": collector.input_tokens,
            "output_tokens": collector.output_tokens,
            "finish_reason": collector.finish_reason,
        }
        if trace_id:
            payload["trace_id"] = trace_id
        if invocation_id:
            payload["invocation_id"] = invocation_id
        agent_name = os.environ.get("DISPATCH_AGENT_NAME")
        if agent_name:
            payload["agent_name"] = agent_name

        log_url = _get_backend_log_url()
        async with httpx.AsyncClient(timeout=10.0) as log_client:
            await log_client.post(log_url, json=payload, headers=_get_auth_headers())
    except Exception:
        logger.debug("Failed to log fallback streaming LLM call", exc_info=True)


# =============================================================================
# Catch-all passthrough — credential injection only (proxy → backend /llm/passthrough)
# =============================================================================


async def _proxy_passthrough(request: Request, provider_format: str) -> Response:
    """Forward an unsupported SDK endpoint to the backend for credential injection.

    Unlike _proxy_to_backend(), this sends the raw request path and method
    to the backend's /llm/passthrough endpoint, which resolves credentials
    and forwards opaquely to the provider. No response parsing or cost tracking.

    Falls back to direct provider call if backend has no config.
    """
    path = "/v1/" + request.path_params["path"]
    method = request.method
    query_string = request.url.query or ""

    # Read raw body (may be empty for GET requests)
    raw_body = await request.body()
    body_dict: dict[str, Any] | None = None
    if raw_body:
        try:
            body_dict = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Non-JSON body in passthrough request, forwarding as-is")

    _, trace_id, invocation_id, subprocess_id, extra_headers = _extract_trace_context(
        request
    )

    # Check fallback cache
    if trace_id and trace_id in _fallback_traces:
        return await _call_provider_passthrough(
            provider_format,
            path,
            method,
            raw_body if raw_body else None,
            query_string,
            extra_headers=extra_headers,
        )

    # Build passthrough request for backend
    backend_payload: dict[str, Any] = {
        "provider_format": provider_format,
        "path": path,
        "method": method,
    }
    if body_dict is not None:
        backend_payload["body"] = body_dict
    if query_string:
        # Parse query string into dict for the backend
        from urllib.parse import parse_qs

        qs = parse_qs(query_string, keep_blank_values=True)
        backend_payload["query_params"] = {k: v[0] for k, v in qs.items()}
    if trace_id:
        backend_payload["trace_id"] = trace_id
    if invocation_id:
        backend_payload["invocation_id"] = invocation_id
    if subprocess_id:
        backend_payload["subprocess_id"] = subprocess_id
    if extra_headers:
        backend_payload["extra_headers"] = extra_headers

    agent_name = os.environ.get("DISPATCH_AGENT_NAME")
    if agent_name:
        backend_payload["agent_name"] = agent_name

    backend_url = _get_backend_passthrough_url()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                backend_url,
                json=backend_payload,
                headers=_get_auth_headers(),
            )
    except (httpx.RequestError, httpcore.ConnectError, OSError):
        # Fall back to direct provider call
        logger.info(
            "Cannot reach backend for passthrough, trying direct %s call",
            provider_format,
        )
        return await _call_provider_passthrough(
            provider_format,
            path,
            method,
            raw_body if raw_body else None,
            query_string,
            extra_headers=extra_headers,
        )

    # Detect "not configured" and fall back to direct provider call
    if _is_not_configured_error(resp.status_code, resp.content):
        logger.info(
            "Backend has no LLM provider configured, falling back to direct %s passthrough",
            provider_format,
        )
        if trace_id:
            _fallback_traces.add(trace_id)
        return await _call_provider_passthrough(
            provider_format,
            path,
            method,
            raw_body if raw_body else None,
            query_string,
            extra_headers=extra_headers,
        )

    # Detect provider auth error — fall back to agent's own key if available
    if _is_auth_error(resp.status_code):
        config = _PROVIDER_CONFIG.get(provider_format)
        if config and os.environ.get(config["key_env"]):
            logger.warning(
                "Provider auth failed (HTTP %d), falling back to agent's own %s key for passthrough",
                resp.status_code,
                provider_format,
            )
            if trace_id:
                _fallback_traces.add(trace_id)
            return await _call_provider_passthrough(
                provider_format,
                path,
                method,
                raw_body if raw_body else None,
                query_string,
                extra_headers=extra_headers,
            )
        logger.warning(
            "Provider auth failed (HTTP %d) and no fallback %s key for passthrough",
            resp.status_code,
            provider_format,
        )

    # Return the raw response from backend (which is the raw provider response)
    content_type = resp.headers.get("content-type", "application/json")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=content_type,
    )


# =============================================================================
# Route handlers
# =============================================================================

# --- OpenAI: supported endpoints (full observability) ---


async def openai_chat(request: Request) -> Response:
    """OpenAI /v1/chat/completions — full observability via backend proxy."""
    return await _proxy_to_backend(request, "openai", "/v1/chat/completions")


async def openai_responses(request: Request) -> Response:
    """OpenAI Responses API /v1/responses — full observability via backend proxy."""
    return await _proxy_to_backend(request, "openai", "/v1/responses")


# --- OpenAI: catch-all (passthrough with credential injection) ---


async def openai_passthrough(request: Request) -> Response:
    """OpenAI catch-all — credential injection via backend passthrough."""
    return await _proxy_passthrough(request, "openai")


# --- Anthropic: supported endpoints (full observability) ---


async def anthropic_messages(request: Request) -> Response:
    """Anthropic /v1/messages — full observability via backend proxy."""
    return await _proxy_to_backend(request, "anthropic", "/v1/messages")


# --- Anthropic: catch-all (passthrough with credential injection) ---


async def anthropic_passthrough(request: Request) -> Response:
    """Anthropic catch-all — credential injection via backend passthrough."""
    return await _proxy_passthrough(request, "anthropic")


def create_app() -> Starlette:
    """Create the sidecar proxy Starlette application.

    Routes are ordered so specific endpoints match before catch-all.
    """
    routes = [
        Route("/health", health, methods=["GET"]),
        # --- OpenAI: supported endpoints (full observability) ---
        Route("/openai/v1/chat/completions", openai_chat, methods=["POST"]),
        Route("/openai/v1/responses", openai_responses, methods=["POST"]),
        # --- OpenAI: catch-all (passthrough) ---
        Route(
            "/openai/v1/{path:path}",
            openai_passthrough,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        ),
        # --- Anthropic: supported endpoints (full observability) ---
        Route("/anthropic/v1/messages", anthropic_messages, methods=["POST"]),
        # --- Anthropic: catch-all (passthrough) ---
        Route(
            "/anthropic/v1/{path:path}",
            anthropic_passthrough,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        ),
    ]
    app = Starlette(routes=routes)
    logger.info(
        "LLM sidecar proxy configured → %s",
        _get_backend_proxy_url(),
    )
    return app


def run_server(port: int = 8780, **_: Any) -> None:
    """Run the sidecar proxy server (blocking). Called from subprocess."""
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )

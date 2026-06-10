"""Internal backend URL and authentication helpers."""

from __future__ import annotations

import os

from dispatch_agents._internal.version import get_sdk_version
from dispatch_agents.config import config as _config


def get_router_url() -> str:
    """Get the dispatch router URL from environment or default."""
    return os.getenv("DISPATCH_BACKEND_URL", "http://dispatch.api:8000")


def get_namespace() -> str | None:
    """Get the dispatch namespace from the runtime config."""
    return _config.namespace


def is_local_dev_mode() -> bool:
    """Return True when running under ``dispatch agent dev``.

    Gated on an explicit ``DISPATCH_LOCAL_DEV`` opt-in rather than heuristics:
    localstack Docker containers should behave like production agents, not dev
    mode. In local dev the router exposes its API non-namespaced, so SDK calls
    must skip the ``/namespace/{ns}`` prefix that deployed agents use.
    """
    return os.getenv("DISPATCH_LOCAL_DEV", "").lower() in ("1", "true", "yes")


def build_api_base_url(backend_url: str, *, namespace: str | None) -> str:
    """Join a backend root with the API path and optional namespace segment.

    The endpoints are non-namespaced in local dev (the dev router only serves
    ``/api/unstable``) or when no ``namespace`` is given; otherwise they are
    scoped under ``/namespace/{namespace}`` as the deployed backend expects.
    """
    base = f"{backend_url}/api/unstable"
    if is_local_dev_mode() or not namespace:
        return base
    return f"{base}/namespace/{namespace}"


def get_api_base_url() -> str:
    """Get the API base URL for SDK→backend data-plane calls."""
    namespace = get_namespace()
    if not is_local_dev_mode() and not namespace:
        raise RuntimeError(
            "DISPATCH_NAMESPACE environment variable is required. "
            "Set it to the namespace your agent is deployed in."
        )
    return build_api_base_url(get_router_url(), namespace=namespace)


def get_auth_headers() -> dict[str, str]:
    """Get authentication and version headers for API requests."""
    headers = {
        "x-dispatch-client": "sdk",
        "x-dispatch-client-version": get_sdk_version(),
        "x-dispatch-client-commit": os.getenv("GIT_COMMIT", "unknown")[:8],
    }

    api_key = os.getenv("DISPATCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers

"""Internal local-development safety helpers."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from dispatch_agents.storage import DisallowedWriteError
from dispatch_agents.storage import get_data_dir as _get_data_dir

_audit_hook_blocked: set[str] = set()
_audit_hook_allowed_prefixes: list[str] = []


def _resolve_path(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except (OSError, ValueError):
        return path


def _init_allowed_prefixes() -> list[str]:
    """Compute the list of allowed write prefixes for dev mode."""
    allowed = [
        _resolve_path(str(_get_data_dir())),
        _resolve_path("/tmp"),
        _resolve_path("/var/tmp"),
        _resolve_path("/private/tmp"),
        _resolve_path("/private/var/tmp"),
        _resolve_path(tempfile.gettempdir()),
    ]

    dev_data_dir = os.environ.get("DISPATCH_DEV_DATA_DIR", "")
    if dev_data_dir:
        agent_folder = str(Path(dev_data_dir).parent.parent)
        allowed.append(_resolve_path(agent_folder))

    return allowed


def _dev_mode_audit_hook(event: str, args: tuple) -> None:
    """Block file operations that target paths outside allowed directories."""
    if event != "open":
        return

    if not args or len(args) < 2:
        return

    raw_path = str(args[0])
    mode = args[1] if len(args) > 1 else ""

    if isinstance(mode, str):
        if "w" not in mode and "a" not in mode and "x" not in mode and "+" not in mode:
            return
    elif isinstance(mode, int):
        write_flags = getattr(os, "O_WRONLY", 1) | getattr(os, "O_RDWR", 2)
        if not (mode & write_flags):
            return

    path = _resolve_path(raw_path)

    for prefix in _audit_hook_allowed_prefixes:
        if path.startswith(prefix):
            return

    show_details = path not in _audit_hook_blocked
    _audit_hook_blocked.add(path)

    if show_details:
        allowed_locations = ", ".join(_audit_hook_allowed_prefixes[:3])
        raise DisallowedWriteError(
            f"Write operation to '{path}' blocked - outside allowed directories.\n"
            f"In dev mode, writes are only allowed to: {allowed_locations}\n"
            "Use your dispatch.yaml volume mount path for persistent storage.\n"
            "To disable this check, run with: dispatch agent dev --allow-arbitrary-writes"
        )

    raise DisallowedWriteError(f"Write to '{path}' blocked (repeated attempt)")


def install_dev_mode_audit_hook() -> None:
    """Install the dev-mode write guard when local dev mode is active."""
    global _audit_hook_allowed_prefixes
    if os.environ.get("DISPATCH_DEV_DATA_DIR") and not os.environ.get(
        "DISPATCH_ALLOW_ARBITRARY_WRITES"
    ):
        _audit_hook_allowed_prefixes = _init_allowed_prefixes()
        sys.addaudithook(_dev_mode_audit_hook)

"""Persistent storage helpers for Dispatch agents."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DisallowedWriteError", "get_data_dir"]


class DisallowedWriteError(Exception):
    """Raised when an agent writes outside allowed directories in dev mode."""


def get_data_dir() -> Path:
    """Return the persistent data directory used by the runtime.

    In production this is the ``/data`` EFS mount. In local dev mode
    (``dispatch agent dev``), it is the mock data directory under
    ``DISPATCH_DEV_DATA_DIR``.

    Use this helper instead of hardcoding ``/data`` so code works in both
    deployed agents and local dev mode.

    Example::

        from dispatch_agents.storage import get_data_dir

        path = get_data_dir() / "state.json"
        path.write_text("{}")
    """
    dev_data_dir = os.environ.get("DISPATCH_DEV_DATA_DIR")
    if dev_data_dir:
        return Path(dev_data_dir) / "data"
    return Path("/data")

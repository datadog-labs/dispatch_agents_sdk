"""Version information for the Dispatch Agents SDK.

This module provides access to version information from pyproject.toml.
"""

import tomllib
from pathlib import Path


def get_sdk_version() -> str:
    """Get SDK version from pyproject.toml (single source of truth)."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data["project"]["version"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
        # Fallback to hardcoded version if file reading fails
        return "unk"

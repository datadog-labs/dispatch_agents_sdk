"""Shared pytest fixtures for the SDK test suite."""

from __future__ import annotations

import pytest

from dispatch_agents.config import _load_runtime_config


@pytest.fixture(autouse=True)
def _clear_runtime_config_cache():
    """Reset the cached runtime config around every test.

    ``dispatch_agents.config`` reads ``namespace``/``agent_name`` through an
    ``lru_cache``d loader, so tests that ``monkeypatch.setenv`` these values
    would otherwise see a stale config cached by an earlier test.
    """
    _load_runtime_config.cache_clear()
    yield
    _load_runtime_config.cache_clear()

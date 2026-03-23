"""Tests for the memory SDK client."""

import pytest

from dispatch_agents.memory import _get_agent_name


class TestGetAgentName:
    """Tests for _get_agent_name helper."""

    def test_explicit_agent_name(self) -> None:
        assert _get_agent_name("my-agent") == "my-agent"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISPATCH_AGENT_NAME", "env-agent")
        assert _get_agent_name() == "env-agent"

    def test_raises_when_no_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISPATCH_AGENT_NAME", raising=False)
        with pytest.raises(ValueError, match="agent_name not provided"):
            _get_agent_name()

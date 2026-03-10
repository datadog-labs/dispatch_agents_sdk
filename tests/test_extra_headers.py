"""Tests for extra_headers ContextVar, context manager, and instrumentation."""

import json

from dispatch_agents.llm import (
    _extra_llm_headers,
    extra_headers,
    get_extra_llm_headers,
)


class TestExtraHeadersContextManager:
    """Test the extra_headers() context manager and ContextVar."""

    def test_default_is_empty(self) -> None:
        """get_extra_llm_headers() returns {} when no context manager is active."""
        assert get_extra_llm_headers() == {}

    def test_sets_headers(self) -> None:
        """Headers are available inside the context manager."""
        with extra_headers({"X-Dataset-Id": "team-ml"}):
            assert get_extra_llm_headers() == {"X-Dataset-Id": "team-ml"}

    def test_resets_on_exit(self) -> None:
        """Headers are cleaned up after the context manager exits."""
        with extra_headers({"X-Foo": "bar"}):
            assert get_extra_llm_headers() == {"X-Foo": "bar"}
        assert get_extra_llm_headers() == {}

    def test_nested_contexts_merge(self) -> None:
        """Inner context merges with outer, inner keys override outer."""
        with extra_headers({"X-Org": "dd", "X-Team": "ml"}):
            assert get_extra_llm_headers() == {"X-Org": "dd", "X-Team": "ml"}
            with extra_headers({"X-Team": "platform", "X-New": "val"}):
                result = get_extra_llm_headers()
                assert result == {
                    "X-Org": "dd",
                    "X-Team": "platform",
                    "X-New": "val",
                }
            # Outer context restored
            assert get_extra_llm_headers() == {"X-Org": "dd", "X-Team": "ml"}
        assert get_extra_llm_headers() == {}

    def test_resets_on_exception(self) -> None:
        """Headers are cleaned up even if an exception occurs inside the block."""
        try:
            with extra_headers({"X-Fail": "yes"}):
                assert get_extra_llm_headers() == {"X-Fail": "yes"}
                raise ValueError("boom")
        except ValueError:
            pass
        assert get_extra_llm_headers() == {}


class TestInstrumentationIntegration:
    """Test that _get_context_headers() picks up extra LLM headers."""

    def test_includes_extra_headers_when_set(self) -> None:
        """X-Dispatch-Extra-Headers is included when ContextVar is non-empty."""
        from dispatch_agents.instrument import _get_context_headers

        with extra_headers({"X-Dataset-Id": "abc"}):
            headers = _get_context_headers()
            assert "X-Dispatch-Extra-Headers" in headers
            parsed = json.loads(headers["X-Dispatch-Extra-Headers"])
            assert parsed == {"X-Dataset-Id": "abc"}

    def test_omits_extra_headers_when_empty(self) -> None:
        """X-Dispatch-Extra-Headers is NOT included when ContextVar is empty."""
        from dispatch_agents.instrument import _get_context_headers

        # Ensure no extra headers are set
        token = _extra_llm_headers.set({})
        try:
            headers = _get_context_headers()
            assert "X-Dispatch-Extra-Headers" not in headers
        finally:
            _extra_llm_headers.reset(token)

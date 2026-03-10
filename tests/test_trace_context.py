"""Unit tests for trace context and fallback lookup functionality."""

import asyncio

import pytest

from dispatch_agents import BasePayload
from dispatch_agents.events import (
    _TRACE_CONTEXT_MAX_SIZE,
    HANDLER_METADATA,
    REGISTERED_HANDLERS,
    TOPIC_HANDLERS,
    _current_invocation_id,
    _current_trace_id,
    _register_trace_invocation,
    _trace_invocation_context,
    _unregister_trace_invocation,
    dispatch_message,
    get_current_invocation_id,
    get_current_trace_id,
    get_invocation_id_for_trace,
    on,
)
from dispatch_agents.mcp import _build_trace_meta
from dispatch_agents.models import ErrorPayload, SuccessPayload, TopicMessage


class TestTraceContextVariables:
    """Tests for context variable operations."""

    def test_get_current_trace_id_default(self):
        """Test default trace_id is None."""
        # Reset context for clean test
        _current_trace_id.set(None)
        assert get_current_trace_id() is None

    def test_get_current_invocation_id_default(self):
        """Test default invocation_id is None."""
        # Reset context for clean test
        _current_invocation_id.set(None)
        assert get_current_invocation_id() is None

    def test_get_current_trace_id_set(self):
        """Test getting trace_id when set."""
        test_trace_id = "trace-123"
        _current_trace_id.set(test_trace_id)
        try:
            assert get_current_trace_id() == test_trace_id
        finally:
            _current_trace_id.set(None)

    def test_get_current_invocation_id_set(self):
        """Test getting invocation_id when set."""
        test_invocation_id = "inv-456"
        _current_invocation_id.set(test_invocation_id)
        try:
            assert get_current_invocation_id() == test_invocation_id
        finally:
            _current_invocation_id.set(None)


class TestTraceInvocationLookup:
    """Tests for the trace -> invocation fallback lookup."""

    def setup_method(self):
        """Clear the context store before each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)

    def teardown_method(self):
        """Clean up after each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)

    def test_lookup_returns_none_for_unknown_trace(self):
        """Test that looking up unknown trace_id returns None."""
        assert get_invocation_id_for_trace("unknown-trace") is None

    def test_lookup_returns_none_for_none_trace(self):
        """Test that looking up None trace_id returns None."""
        assert get_invocation_id_for_trace(None) is None

    def test_lookup_returns_stored_invocation_id(self):
        """Test that stored invocation_id can be looked up."""
        trace_id = "trace-abc"
        invocation_id = "inv-xyz"
        _register_trace_invocation(trace_id, invocation_id)

        assert get_invocation_id_for_trace(trace_id) == invocation_id

    def test_lookup_handles_multiple_traces(self):
        """Test that multiple trace -> invocation mappings work."""
        _register_trace_invocation("trace-1", "inv-1")
        _register_trace_invocation("trace-2", "inv-2")
        _register_trace_invocation("trace-3", "inv-3")

        assert get_invocation_id_for_trace("trace-1") == "inv-1"
        assert get_invocation_id_for_trace("trace-2") == "inv-2"
        assert get_invocation_id_for_trace("trace-3") == "inv-3"


class TestBuildTraceMeta:
    """Tests for _build_trace_meta with fallback lookup."""

    def setup_method(self):
        """Clear context before each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)

    def teardown_method(self):
        """Clean up after each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)

    def test_returns_none_when_no_context(self):
        """Test that None is returned when no trace context is available."""
        result = _build_trace_meta()
        assert result is None

    def test_returns_both_when_context_set(self):
        """Test that both trace_id and invocation_id are included when set."""
        _current_trace_id.set("trace-123")
        _current_invocation_id.set("inv-456")

        result = _build_trace_meta()

        assert result == {
            "dispatch_trace_id": "trace-123",
            "dispatch_invocation_id": "inv-456",
        }

    def test_returns_only_trace_id_when_invocation_missing_and_no_fallback(self):
        """Test that only trace_id is returned when invocation_id not available."""
        _current_trace_id.set("trace-123")
        # No invocation_id set, and no fallback registered

        result = _build_trace_meta()

        assert result == {"dispatch_trace_id": "trace-123"}

    def test_uses_fallback_when_invocation_id_missing(self):
        """Test that fallback lookup is used when invocation_id not in context."""
        # Set trace_id in context variable
        _current_trace_id.set("trace-123")
        # Don't set invocation_id in context variable
        # But register it in the fallback store
        _register_trace_invocation("trace-123", "fallback-inv-456")

        result = _build_trace_meta()

        assert result == {
            "dispatch_trace_id": "trace-123",
            "dispatch_invocation_id": "fallback-inv-456",
        }

    def test_context_variable_takes_precedence_over_fallback(self):
        """Test that context variable is preferred over fallback lookup."""
        _current_trace_id.set("trace-123")
        _current_invocation_id.set("context-inv-789")
        # Also register a different value in fallback
        _register_trace_invocation("trace-123", "fallback-inv-456")

        result = _build_trace_meta()

        # Should use context variable, not fallback
        assert result == {
            "dispatch_trace_id": "trace-123",
            "dispatch_invocation_id": "context-inv-789",
        }

    def test_fallback_not_used_when_trace_id_missing(self):
        """Test that fallback isn't used when trace_id itself is missing."""
        # Don't set trace_id, only register fallback
        _register_trace_invocation("trace-123", "inv-456")

        result = _build_trace_meta()

        assert result is None


class TestTraceContextBoundedSize:
    """Tests for bounded size and garbage collection of trace context store."""

    def setup_method(self):
        """Clear the context store before each test."""
        _trace_invocation_context.clear()

    def teardown_method(self):
        """Clean up after each test."""
        _trace_invocation_context.clear()

    def test_max_size_constant_is_reasonable(self):
        """Test that max size is set to a reasonable value."""
        assert _TRACE_CONTEXT_MAX_SIZE == 100000

    def test_entries_evicted_when_over_limit(self, monkeypatch):
        """Test that oldest entries are evicted when cache exceeds max size."""
        # Use a smaller size for testing
        import dispatch_agents.events as events_module

        test_max_size = 100
        monkeypatch.setattr(events_module, "_TRACE_CONTEXT_MAX_SIZE", test_max_size)

        # Fill the cache to max size
        for i in range(test_max_size):
            _register_trace_invocation(f"trace-{i}", f"inv-{i}")

        assert len(_trace_invocation_context) == test_max_size

        # First entry should still be present
        assert get_invocation_id_for_trace("trace-0") == "inv-0"

        # Add one more entry - should evict the oldest
        _register_trace_invocation("trace-new", "inv-new")

        # Size should still be at max
        assert len(_trace_invocation_context) == test_max_size

        # First entry should be evicted
        assert get_invocation_id_for_trace("trace-0") is None

        # Second entry should still be present
        assert get_invocation_id_for_trace("trace-1") == "inv-1"

        # New entry should be present
        assert get_invocation_id_for_trace("trace-new") == "inv-new"

    def test_multiple_evictions(self, monkeypatch):
        """Test that multiple entries can be evicted correctly."""
        # Use a smaller size for testing
        import dispatch_agents.events as events_module

        test_max_size = 100
        monkeypatch.setattr(events_module, "_TRACE_CONTEXT_MAX_SIZE", test_max_size)

        # Fill the cache
        for i in range(test_max_size):
            _register_trace_invocation(f"trace-{i}", f"inv-{i}")

        # Add 10 more entries
        for i in range(10):
            _register_trace_invocation(f"trace-extra-{i}", f"inv-extra-{i}")

        # Size should still be at max
        assert len(_trace_invocation_context) == test_max_size

        # First 10 entries should be evicted
        for i in range(10):
            assert get_invocation_id_for_trace(f"trace-{i}") is None

        # Entry 10 should still be present
        assert get_invocation_id_for_trace("trace-10") == "inv-10"

        # All extra entries should be present
        for i in range(10):
            assert get_invocation_id_for_trace(f"trace-extra-{i}") == f"inv-extra-{i}"

    def test_updating_existing_entry_moves_to_end(self, monkeypatch):
        """Test that updating an existing entry moves it to most-recently-used."""
        # Use a smaller size for testing
        import dispatch_agents.events as events_module

        test_max_size = 100
        monkeypatch.setattr(events_module, "_TRACE_CONTEXT_MAX_SIZE", test_max_size)

        # Fill the cache
        for i in range(test_max_size):
            _register_trace_invocation(f"trace-{i}", f"inv-{i}")

        # Update the first entry (should move to end, making it "newest")
        _register_trace_invocation("trace-0", "inv-0-updated")

        # Add a new entry - should evict trace-1 (now the oldest), not trace-0
        _register_trace_invocation("trace-new", "inv-new")

        # trace-0 should still be present (was moved to end)
        assert get_invocation_id_for_trace("trace-0") == "inv-0-updated"

        # trace-1 should be evicted (was oldest after trace-0 was updated)
        assert get_invocation_id_for_trace("trace-1") is None

    def test_clear_empties_cache(self):
        """Test that clearing the cache removes all entries."""
        for i in range(100):
            _register_trace_invocation(f"trace-{i}", f"inv-{i}")

        assert len(_trace_invocation_context) == 100

        _trace_invocation_context.clear()

        assert len(_trace_invocation_context) == 0
        assert get_invocation_id_for_trace("trace-0") is None

    def test_unregister_removes_entry(self):
        """Test that unregistering removes the entry."""
        _register_trace_invocation("trace-123", "inv-456")
        assert get_invocation_id_for_trace("trace-123") == "inv-456"

        _unregister_trace_invocation("trace-123")
        assert get_invocation_id_for_trace("trace-123") is None

    def test_unregister_nonexistent_is_safe(self):
        """Test that unregistering a nonexistent entry doesn't raise."""
        # Should not raise
        _unregister_trace_invocation("nonexistent-trace")


class TestDispatchMessageCleanup:
    """Tests for deterministic cleanup when dispatch_message returns."""

    def setup_method(self):
        """Clear registries and context before each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)
        # Save original registries
        self._orig_handlers = dict(REGISTERED_HANDLERS)
        self._orig_topics = dict(TOPIC_HANDLERS)
        self._orig_metadata = dict(HANDLER_METADATA)

    def teardown_method(self):
        """Restore registries and clean up after each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)
        # Restore original registries
        REGISTERED_HANDLERS.clear()
        REGISTERED_HANDLERS.update(self._orig_handlers)
        TOPIC_HANDLERS.clear()
        TOPIC_HANDLERS.update(self._orig_topics)
        HANDLER_METADATA.clear()
        HANDLER_METADATA.update(self._orig_metadata)

    @pytest.mark.asyncio
    async def test_trace_context_cleaned_up_after_successful_dispatch(self):
        """Test that trace context is cleaned up after dispatch_message succeeds."""

        class TestPayload(BasePayload):
            value: str

        @on(topic="test.cleanup.success")
        async def handle_test(payload: TestPayload) -> None:
            # During handler execution, context should be registered
            assert (
                get_invocation_id_for_trace("trace-cleanup-test") == "inv-cleanup-test"
            )
            return None

        message = TopicMessage(
            uid="inv-cleanup-test",
            topic="test.cleanup.success",
            payload={"value": "test"},
            trace_id="trace-cleanup-test",
            parent_id=None,
            sender_id="test",
            ts="2025-01-01T00:00:00Z",
        )

        # Before dispatch, context should not be registered
        assert get_invocation_id_for_trace("trace-cleanup-test") is None

        # Dispatch the message
        result = await dispatch_message(message)

        # After dispatch, context should be cleaned up
        assert get_invocation_id_for_trace("trace-cleanup-test") is None
        assert isinstance(result, SuccessPayload)
        assert result.result is None  # Handler returned None

    @pytest.mark.asyncio
    async def test_trace_context_cleaned_up_after_handler_exception(self):
        """Test that trace context is cleaned up even if handler raises."""

        class TestPayload(BasePayload):
            value: str

        @on(topic="test.cleanup.error")
        async def handle_test_error(payload: TestPayload) -> None:
            raise ValueError("Test error")

        message = TopicMessage(
            uid="inv-cleanup-error",
            topic="test.cleanup.error",
            payload={"value": "test"},
            trace_id="trace-cleanup-error",
            parent_id=None,
            sender_id="test",
            ts="2025-01-01T00:00:00Z",
        )

        # Dispatch the message (handler will raise, but dispatch catches it)
        result = await dispatch_message(message)

        # After dispatch, context should be cleaned up even though handler raised
        assert get_invocation_id_for_trace("trace-cleanup-error") is None
        assert isinstance(result, ErrorPayload)
        assert result.error == "Test error"  # Error should be captured


class TestAsyncContextPropagation:
    """Tests for async context propagation scenarios."""

    def setup_method(self):
        """Clear context before each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)

    def teardown_method(self):
        """Clean up after each test."""
        _trace_invocation_context.clear()
        _current_trace_id.set(None)
        _current_invocation_id.set(None)

    @pytest.mark.asyncio
    async def test_context_propagates_to_child_task(self):
        """Test that context variables propagate to asyncio.create_task()."""
        _current_trace_id.set("trace-parent")
        _current_invocation_id.set("inv-parent")

        async def child_task():
            return get_current_trace_id(), get_current_invocation_id()

        task = asyncio.create_task(child_task())
        trace_id, inv_id = await task

        assert trace_id == "trace-parent"
        assert inv_id == "inv-parent"

    @pytest.mark.asyncio
    async def test_fallback_works_in_separate_context(self):
        """Test that fallback lookup works when context vars aren't propagated.

        This simulates what happens when external SDKs create tasks
        that don't inherit Python context variables.
        """
        # Register the mapping
        _register_trace_invocation("trace-123", "inv-456")

        # Simulate a task that only has trace_id (maybe from headers/other source)
        # but doesn't have invocation_id in context
        async def external_sdk_context():
            # This task doesn't have context variables set
            # But it knows the trace_id from some other source
            _current_trace_id.set("trace-123")
            # Don't set invocation_id - simulating broken context propagation
            return _build_trace_meta()

        result = await external_sdk_context()

        # Should use fallback to get invocation_id
        assert result == {
            "dispatch_trace_id": "trace-123",
            "dispatch_invocation_id": "inv-456",
        }

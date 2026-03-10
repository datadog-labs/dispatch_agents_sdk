"""Tests for dispatch_agents.proxy.sse_utils module."""

import time

from dispatch_agents.proxy.sse_utils import StreamingUsageCollector

# ── StreamingUsageCollector ──────────────────────────────────────────


class TestStreamingUsageCollectorOpenAI:
    def test_extracts_model_and_usage(self):
        collector = StreamingUsageCollector("openai")
        collector.observe({"model": "gpt-4o", "choices": []})
        collector.observe(
            {
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "choices": [{"finish_reason": "stop"}],
            }
        )
        collector.finalize()

        assert collector.model == "gpt-4o"
        assert collector.input_tokens == 10
        assert collector.output_tokens == 20
        assert collector.finish_reason == "stop"
        assert collector.latency_ms >= 0

    def test_handles_empty_choices(self):
        collector = StreamingUsageCollector("openai")
        collector.observe({"choices": []})
        assert collector.finish_reason == "stop"  # default

    def test_ignores_non_dict_non_model(self):
        collector = StreamingUsageCollector("openai")
        collector.observe("not a dict")
        assert collector.model == "unknown"


class TestStreamingUsageCollectorAnthropic:
    def test_extracts_from_message_events(self):
        collector = StreamingUsageCollector("anthropic")
        collector.observe(
            {
                "type": "message_start",
                "message": {
                    "model": "claude-sonnet-4-5-20250929",
                    "usage": {"input_tokens": 15},
                },
            }
        )
        collector.observe(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 25},
            }
        )
        collector.finalize()

        assert collector.model == "claude-sonnet-4-5-20250929"
        assert collector.input_tokens == 15
        assert collector.output_tokens == 25
        assert collector.finish_reason == "stop"
        assert collector.latency_ms >= 0

    def test_stop_reason_mapping(self):
        mappings = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "custom_reason": "custom_reason",  # unmapped passthrough
        }
        for anthropic_reason, expected in mappings.items():
            collector = StreamingUsageCollector("anthropic")
            collector.observe(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": anthropic_reason},
                    "usage": {},
                }
            )
            assert collector.finish_reason == expected, (
                f"Expected {expected!r} for {anthropic_reason!r}"
            )

    def test_finalize_sets_latency(self):
        collector = StreamingUsageCollector("openai")
        time.sleep(0.01)
        collector.finalize()
        assert collector.latency_ms >= 10

"""SSE utilities for LLM streaming responses.

Provides usage extraction for the sidecar proxy's fallback path
(direct-to-provider calls when backend has no config).

For the backend path, the sidecar just passes SSE bytes through without parsing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class StreamingUsageCollector:
    """Accumulates usage data from streaming chunks."""

    def __init__(self, provider_format: str) -> None:
        self.provider_format = provider_format
        self.model: str = "unknown"
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.finish_reason: str = "stop"
        self._start_time = time.time()
        self.latency_ms: int = 0

    def observe(self, chunk: dict[str, Any] | Any) -> None:
        if hasattr(chunk, "model_dump"):
            d = chunk.model_dump()
        elif isinstance(chunk, dict):
            d = chunk
        else:
            return

        if self.provider_format == "anthropic":
            self._observe_anthropic(d)
        else:
            self._observe_openai(d)

    def finalize(self) -> None:
        self.latency_ms = int((time.time() - self._start_time) * 1000)

    def _observe_anthropic(self, d: dict[str, Any]) -> None:
        ev_type = d.get("type")
        if ev_type == "message_start":
            msg = d.get("message", {})
            self.model = msg.get("model", self.model)
            usage = msg.get("usage", {})
            self.input_tokens = usage.get("input_tokens", 0)
        elif ev_type == "message_delta":
            delta = d.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                reason_map = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                }
                self.finish_reason = reason_map.get(stop_reason, stop_reason)
            usage = d.get("usage", {})
            if usage.get("output_tokens"):
                self.output_tokens = usage["output_tokens"]

    def _observe_openai(self, d: dict[str, Any]) -> None:
        if d.get("model"):
            self.model = d["model"]
        usage = d.get("usage")
        if usage:
            self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)
        choices = d.get("choices", [])
        if choices and choices[0].get("finish_reason"):
            self.finish_reason = choices[0]["finish_reason"]

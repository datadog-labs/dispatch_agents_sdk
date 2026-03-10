"""Tests for dispatch_agents.llm_langchain module.

Covers the pure helper functions and response conversion without
needing a running LLM client.
"""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from dispatch_agents.llm import LLMResponse

import pytest

langchain_core = pytest.importorskip(
    "langchain_core", reason="langchain_core not installed"
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from dispatch_agents.llm_langchain import (
    ChatDispatch,
    _convert_message_to_dict,
    _convert_response_to_message,
    _convert_tools_to_openai_format,
    _get_tool_schema,
    _parse_tool_arguments,
    _serialize_tool_arguments,
)

# ── _get_tool_schema ─────────────────────────────────────────────────


class TestGetToolSchema:
    def test_pydantic_model(self):
        tool = SimpleNamespace()
        tool.args_schema = MagicMock()
        tool.args_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        result = _get_tool_schema(tool)
        assert result == {"type": "object", "properties": {"x": {"type": "string"}}}

    def test_dict_schema(self):
        tool = SimpleNamespace()
        tool.args_schema = {"type": "object", "properties": {"y": {"type": "integer"}}}
        result = _get_tool_schema(tool)
        assert result == {"type": "object", "properties": {"y": {"type": "integer"}}}

    def test_no_args_schema(self):
        tool = SimpleNamespace()
        result = _get_tool_schema(tool)
        assert result == {}

    def test_none_args_schema(self):
        tool = SimpleNamespace()
        tool.args_schema = None
        result = _get_tool_schema(tool)
        assert result == {}


# ── _convert_tools_to_openai_format ──────────────────────────────────


class TestConvertToolsToOpenaiFormat:
    def test_langchain_tool(self):
        tool = SimpleNamespace()
        tool.name = "search"
        tool.description = "Search the web"
        tool.args_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
        result = _convert_tools_to_openai_format([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search the web"
        assert result[0]["function"]["parameters"] == tool.args_schema

    def test_dict_passthrough(self):
        tool_dict = {
            "type": "function",
            "function": {"name": "calc", "description": "Calculate"},
        }
        result = _convert_tools_to_openai_format([tool_dict])
        assert result == [tool_dict]

    def test_callable_function(self):
        def my_func(x: str, y: str):
            """Do something useful."""
            pass

        result = _convert_tools_to_openai_format([my_func])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "my_func"
        assert result[0]["function"]["description"] == "Do something useful."
        assert "x" in result[0]["function"]["parameters"]["properties"]
        assert "y" in result[0]["function"]["parameters"]["properties"]

    def test_tool_with_no_params(self):
        tool = SimpleNamespace()
        tool.name = "ping"
        tool.description = "Ping the server"
        result = _get_tool_schema(tool)
        assert result == {}

    def test_tool_with_empty_description(self):
        tool = SimpleNamespace()
        tool.name = "noop"
        tool.description = None
        tool.args_schema = None
        result = _convert_tools_to_openai_format([tool])
        assert result[0]["function"]["description"] == ""


# ── _serialize_tool_arguments ────────────────────────────────────────


class TestSerializeToolArguments:
    def test_dict(self):
        result = _serialize_tool_arguments({"a": 1, "b": "two"})
        assert json.loads(result) == {"a": 1, "b": "two"}

    def test_string_passthrough(self):
        result = _serialize_tool_arguments('{"key": "val"}')
        assert result == '{"key": "val"}'

    def test_empty_dict(self):
        result = _serialize_tool_arguments({})
        assert result == "{}"

    def test_unconvertible_fallback(self):
        result = _serialize_tool_arguments(42)
        # int(42) can't be converted to dict
        assert result == "{}"


# ── _convert_message_to_dict ─────────────────────────────────────────


class TestConvertMessageToDict:
    def test_human_message(self):
        msg = HumanMessage(content="hi")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "user", "content": "hi"}

    def test_ai_message(self):
        msg = AIMessage(content="hello")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "assistant", "content": "hello"}

    def test_system_message(self):
        msg = SystemMessage(content="You are helpful.")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "system", "content": "You are helpful."}

    def test_tool_message(self):
        msg = ToolMessage(content="result", tool_call_id="call_123")
        result = _convert_message_to_dict(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "result"

    def test_tool_message_no_call_id(self):
        msg = ToolMessage(content="result", tool_call_id="")
        result = _convert_message_to_dict(msg)
        assert result["role"] == "tool"
        # Should use fallback id
        assert result["tool_call_id"]

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search", "args": {"query": "hello"}},
            ],
        )
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "hello"}


# ── _parse_tool_arguments ────────────────────────────────────────────


class TestParseToolArguments:
    def test_dict_passthrough(self):
        assert _parse_tool_arguments({"a": 1}) == {"a": 1}

    def test_json_string(self):
        assert _parse_tool_arguments('{"a": 1}') == {"a": 1}

    def test_empty_string(self):
        assert _parse_tool_arguments("") == {}

    def test_whitespace_string(self):
        assert _parse_tool_arguments("   ") == {}

    def test_invalid_json(self):
        assert _parse_tool_arguments("not json") == {}

    def test_non_dict_json(self):
        assert _parse_tool_arguments("[1,2,3]") == {}

    def test_none(self):
        assert _parse_tool_arguments(None) == {}

    def test_int(self):
        assert _parse_tool_arguments(42) == {}


# ── _convert_response_to_message ─────────────────────────────────────


class TestConvertResponseToMessage:
    def _make_response(self, **overrides: Any) -> "LLMResponse":
        from dispatch_agents.llm import LLMResponse

        return LLMResponse(
            llm_call_id=overrides.get("llm_call_id", "call-123"),
            content=overrides.get("content", "Hello!"),
            tool_calls=overrides.get("tool_calls", None),
            finish_reason=overrides.get("finish_reason", "stop"),
            model=overrides.get("model", "gpt-4o"),
            provider=overrides.get("provider", "openai"),
            variant_name=overrides.get("variant_name", None),
            input_tokens=overrides.get("input_tokens", 10),
            output_tokens=overrides.get("output_tokens", 5),
            cost_usd=overrides.get("cost_usd", 0.001),
            latency_ms=overrides.get("latency_ms", 200),
        )

    def test_simple_response(self):
        resp = self._make_response()
        msg = _convert_response_to_message(resp)

        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello!"
        assert msg.response_metadata["model"] == "gpt-4o"
        assert msg.response_metadata["provider"] == "openai"
        assert msg.response_metadata["finish_reason"] == "stop"
        assert msg.response_metadata["llm_call_id"] == "call-123"
        assert msg.usage_metadata is not None
        assert msg.usage_metadata["input_tokens"] == 10
        assert msg.usage_metadata["output_tokens"] == 5
        assert msg.usage_metadata["total_tokens"] == 15

    def test_response_with_tool_calls(self):
        from dispatch_agents.llm import LLMFunctionCall, LLMToolCall

        tc = LLMToolCall(
            id="call_abc",
            function=LLMFunctionCall(name="search", arguments='{"q": "test"}'),
        )
        resp = self._make_response(content=None, tool_calls=[tc])
        msg = _convert_response_to_message(resp)

        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "search"
        assert msg.tool_calls[0]["args"] == {"q": "test"}
        assert msg.tool_calls[0]["id"] == "call_abc"

    def test_variant_name_in_metadata(self):
        resp = self._make_response(variant_name="test-variant")
        msg = _convert_response_to_message(resp)
        assert msg.response_metadata["variant_name"] == "test-variant"

    def test_no_variant_name(self):
        resp = self._make_response(variant_name=None)
        msg = _convert_response_to_message(resp)
        assert "variant_name" not in msg.response_metadata


# ── ChatDispatch ─────────────────────────────────────────────────────


class TestChatDispatch:
    @patch("dispatch_agents.llm_langchain.LLMClient")
    def test_llm_type(self, mock_client_cls):
        llm = ChatDispatch()
        assert llm._llm_type == "dispatch"

    @patch("dispatch_agents.llm_langchain.LLMClient")
    def test_identifying_params(self, mock_client_cls):
        llm = ChatDispatch(
            model="gpt-4o", provider="openai", temperature=0.5, max_tokens=100
        )
        params = llm._identifying_params
        assert params["model"] == "gpt-4o"
        assert params["provider"] == "openai"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100

    @patch("dispatch_agents.llm_langchain.LLMClient")
    def test_default_params(self, mock_client_cls):
        llm = ChatDispatch()
        params = llm._identifying_params
        assert params["model"] is None
        assert params["provider"] is None
        assert params["temperature"] == 1.0
        assert params["max_tokens"] is None

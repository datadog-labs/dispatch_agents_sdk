"""LLM inference client for Dispatch agents.

The LLM helpers call providers through the Dispatch proxy and automatically
correlate calls with the current agent invocation trace when used inside
``@fn`` or ``@on`` handlers.

LLM calls should be made inside handler functions. Calls made at module import
time are not associated with a Dispatch invocation trace.

Example::

    from dispatch_agents import BasePayload, fn, llm
    from dispatch_agents.llm import parse_json
    from dispatch_agents.models import LLMMessage

    class Analysis(BasePayload):
        sentiment: str
        confidence: float

    class TextRequest(BasePayload):
        text: str

    class ChatRequest(BasePayload):
        prompt: str

    @fn()
    async def analyze(payload: TextRequest) -> Analysis:
        response = await llm.chat(
            payload.text,
            system="Analyze sentiment and return JSON.",
            response_format=Analysis,
        )
        return parse_json(response, Analysis)

    @fn()
    async def conversation(payload: ChatRequest) -> str | None:
        response = await llm.inference([
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content=payload.prompt),
        ])
        return response.content
"""

from __future__ import annotations

import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypedDict, TypeVar, overload

import httpx

from dispatch_agents._internal.dispatch import (
    get_current_invocation_id as _get_current_invocation_id,
)
from dispatch_agents._internal.dispatch import (
    get_current_trace_id as _get_current_trace_id,
)
from dispatch_agents._internal.transport import get_api_base_url as _get_api_base_url
from dispatch_agents._internal.transport import get_auth_headers as _get_auth_headers
from dispatch_agents.config import config as _config
from dispatch_agents.models import BasePayload as _BasePayload
from dispatch_agents.models import JsonObject as _JsonObject
from dispatch_agents.models import JsonValue as _JsonValue
from dispatch_agents.models import LLMMessage as _LLMMessage
from dispatch_agents.models import LLMResponse as _LLMResponse
from dispatch_agents.models import LLMToolCall as _LLMToolCall

__all__ = [
    "LLMClient",
    "chat",
    "extra_headers",
    "get_extra_llm_headers",
    "inference",
    "llm",
    "log_anthropic_response",
    "log_llm_call",
    "log_openai_response",
    "log_response",
    "parse_json",
]

_extra_llm_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "extra_llm_headers", default=None
)


@contextmanager
def extra_headers(headers: dict[str, str]) -> Generator[None, None, None]:
    """Attach extra headers to LLM provider requests in the current context.

    Headers are forwarded through the Dispatch proxy to the underlying LLM
    provider. Nested contexts merge with outer contexts; inner keys override
    outer keys.

    Example::

        from dispatch_agents import extra_headers, fn, llm

        @fn()
        async def answer(payload: Question) -> str | None:
            with extra_headers({"X-Dataset-Id": "team-ml"}):
                response = await llm.chat(payload.question)
            return response.content
    """
    current = _extra_llm_headers.get() or {}
    merged = {**current, **headers}
    token = _extra_llm_headers.set(merged)
    try:
        yield
    finally:
        _extra_llm_headers.reset(token)


def get_extra_llm_headers() -> dict[str, str]:
    """Return the current extra LLM headers, or an empty dict when none are set."""
    return _extra_llm_headers.get() or {}


def _to_json_value(value: object) -> _JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode(errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_to_json_value(item) for item in value]
    return str(value)


def _to_json_object(value: object) -> _JsonObject:
    json_value = _to_json_value(value)
    if not isinstance(json_value, dict):
        raise ValueError("Expected JSON object")
    return json_value


# Narrow the return type to the caller's model when ``model`` is supplied
# (e.g. ``parse_json(resp, Colors) -> Colors``); otherwise it's parsed JSON.
_ResponseT = TypeVar("_ResponseT", bound=_BasePayload)


@overload
def parse_json(response: _LLMResponse, model: type[_ResponseT]) -> _ResponseT: ...


@overload
def parse_json(response: _LLMResponse, model: None = None) -> _JsonValue: ...


def parse_json(
    response: _LLMResponse,
    model: type[_BasePayload] | None = None,
) -> _BasePayload | _JsonValue:
    """Parse an LLM response's JSON content.

    Args:
        response: LLM response returned by :func:`chat` or :func:`inference`.
        model: Optional :class:`BasePayload` subclass used to validate and parse
            the JSON content.

    Returns:
        Parsed JSON value when ``model`` is omitted, otherwise an instance of
        ``model``.

    Raises:
        ValueError: If the response has no content.
        json.JSONDecodeError: If the content is not valid JSON.
        pydantic.ValidationError: If ``model`` validation fails.

    Example::

        from dispatch_agents import BasePayload
        from dispatch_agents.llm import chat, parse_json

        class Colors(BasePayload):
            colors: list[str]

        response = await chat("Return JSON with three primary colors.", response_format=Colors)
        result = parse_json(response, Colors)
        print(result.colors)
    """
    if not response.content:
        raise ValueError("Response has no content to parse")

    loaded: object = json.loads(response.content)
    data = _to_json_value(loaded)
    if model is not None:
        return model.model_validate(data)
    return data


def _required_str(data: _JsonObject, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"LLM response missing string field '{key}'")
    return value


def _optional_str(data: _JsonObject, key: str) -> str | None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"LLM response field '{key}' must be a string")


def _required_int(data: _JsonObject, key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"LLM response missing integer field '{key}'")
    return value


def _required_float(data: _JsonObject, key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"LLM response missing numeric field '{key}'")
    return float(value)


class LLMClient:
    """Client for LLM inference via the Dispatch proxy.

    The client can hold default provider settings while individual calls can
    override them. It automatically propagates Dispatch trace context for
    observability when called inside a handler.

    Example::

        client = LLMClient(provider="openai", model="gpt-4o")
        response = await client.chat(
            "Explain quantum computing",
            system="Explain complex topics simply.",
        )
        print(response.content)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize an LLM client with optional defaults.

        Args:
            model: Default model, such as ``"gpt-4o"`` or ``"claude-sonnet-4-5"``.
            provider: Default provider, such as ``"openai"`` or ``"anthropic"``.
            temperature: Default sampling temperature.
            max_tokens: Default maximum response tokens.
        """
        self._api_base_url: str | None = None
        self._default_model = model
        self._default_provider = provider
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens

    def _ensure_api_base_url(self) -> str:
        if self._api_base_url is None:
            self._api_base_url = _get_api_base_url()
        return self._api_base_url

    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: _JsonObject | type[_BasePayload] | None = None,
    ) -> _LLMResponse:
        """Send a single user message and return the model response.

        Args:
            message: User message to send.
            system: Optional system prompt.
            model: Model to use. Falls back to the client or org default.
            provider: Provider to route to. Falls back to the client or org default.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            response_format: Structured output request. Pass
                ``{"type": "json_object"}`` for JSON mode, or a
                :class:`BasePayload` subclass for schema-guided generation.

        Returns:
            LLM response with content, tool calls, usage, cost, and latency.

        Example::

            response = await llm.chat("What is 2+2?")
            print(response.content)

        Example::

            class Colors(BasePayload):
                colors: list[str]

            response = await llm.chat(
                "List three primary colors as JSON.",
                response_format=Colors,
            )
            colors = parse_json(response, Colors)
        """
        messages: list[_LLMMessage] = []
        if system:
            messages.append(_LLMMessage(role="system", content=system))
        messages.append(_LLMMessage(role="user", content=message))

        format_dict: _JsonObject | None = None
        if response_format is not None:
            if isinstance(response_format, dict):
                format_dict = response_format
            elif isinstance(response_format, type) and issubclass(
                response_format, _BasePayload
            ):
                format_dict = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": _to_json_object(response_format.model_json_schema()),
                    },
                }

        return await self.inference(
            messages,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=format_dict,
        )

    async def inference(
        self,
        messages: Sequence[_LLMMessage],
        *,
        model: str | None = None,
        provider: str | None = None,
        tools: Sequence[_JsonObject] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: _JsonObject | None = None,
        trace_id: str | None = None,
        invocation_id: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> _LLMResponse:
        """Execute LLM inference over a typed message sequence.

        Args:
            messages: Conversation messages as public ``LLMMessage`` models.
            model: Model to use. If omitted, falls back to the provider default.
            provider: Provider to route to. When passing ``model``, pass
                ``provider`` too so the model is sent to the intended provider.
            tools: Tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            response_format: Structured output format, such as
                ``{"type": "json_object"}``.
            trace_id: Optional trace ID override. Auto-detected from handler
                context when omitted.
            invocation_id: Optional invocation ID override. Auto-detected from
                handler context when omitted.
            extra_headers: Extra provider headers for this request.

        Returns:
            LLM response with content, tool calls, usage, cost, and latency.

        Raises:
            httpx.HTTPStatusError: If the LLM proxy rejects the request.
            RuntimeError: If required Dispatch runtime configuration is missing.

        Example::

            from dispatch_agents.models import LLMMessage

            response = await llm.inference([
                LLMMessage(role="system", content="You are helpful."),
                LLMMessage(role="user", content="What is 2+2?"),
            ])
            print(response.content)
        """
        api_base_url = self._ensure_api_base_url()

        messages_payload: list[_JsonValue] = []
        for msg in messages:
            messages_payload.append(_to_json_object(msg.model_dump(exclude_none=True)))

        if trace_id is None:
            trace_id = _get_current_trace_id()
        if invocation_id is None:
            invocation_id = _get_current_invocation_id()

        effective_model = model if model is not None else self._default_model
        effective_provider = (
            provider if provider is not None else self._default_provider
        )
        effective_temperature = (
            temperature if temperature is not None else self._default_temperature
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )

        payload: _JsonObject = {
            "messages": messages_payload,
        }

        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        if effective_model is not None:
            payload["model"] = effective_model
        if effective_provider is not None:
            payload["provider"] = effective_provider
        if tools is not None:
            tools_payload: list[_JsonValue] = []
            tools_payload.extend(tools)
            payload["tools"] = tools_payload
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if invocation_id is not None:
            payload["invocation_id"] = invocation_id

        agent_name = _config.agent_name
        if agent_name:
            payload["agent_name"] = agent_name

        merged_headers: dict[str, _JsonValue] = {**get_extra_llm_headers()}
        if extra_headers:
            merged_headers.update(extra_headers)
        if merged_headers:
            payload["extra_headers"] = merged_headers

        url = f"{api_base_url}/llm/inference"
        auth_headers = _get_auth_headers()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=auth_headers,
                timeout=600.0,
            )
            if response.status_code >= 400:
                try:
                    error_body = response.json()
                    detail = error_body.get("detail", response.text)
                except Exception:
                    detail = response.text
                raise httpx.HTTPStatusError(
                    f"LLM inference failed ({response.status_code}): {detail}",
                    request=response.request,
                    response=response,
                )
            data = _to_json_object(response.json())

        tool_calls = None
        raw_tool_calls = data.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            tool_calls = []
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    raise ValueError("LLM response tool_calls must contain objects")
                tool_calls.append(_LLMToolCall.model_validate(raw_tool_call))

        return _LLMResponse(
            llm_call_id=_required_str(data, "llm_call_id"),
            content=_optional_str(data, "content"),
            tool_calls=tool_calls,
            finish_reason=_required_str(data, "finish_reason"),
            model=_required_str(data, "model"),
            provider=_required_str(data, "provider"),
            variant_name=_optional_str(data, "variant_name"),
            input_tokens=_required_int(data, "input_tokens"),
            output_tokens=_required_int(data, "output_tokens"),
            cost_usd=_required_float(data, "cost_usd"),
            latency_ms=_required_int(data, "latency_ms"),
        )


llm = LLMClient()
"""Singleton LLM client."""


async def chat(
    message: str,
    *,
    system: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: _JsonObject | type[_BasePayload] | None = None,
) -> _LLMResponse:
    """Send a single user message with the singleton LLM client.

    See :meth:`LLMClient.chat` for parameters, return value, and examples.
    """
    return await llm.chat(
        message,
        system=system,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )


async def inference(
    messages: Sequence[_LLMMessage],
    *,
    model: str | None = None,
    provider: str | None = None,
    tools: Sequence[_JsonObject] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: _JsonObject | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> _LLMResponse:
    """Execute LLM inference with the singleton LLM client.

    See :meth:`LLMClient.inference` for parameters, return value, and examples.
    """
    return await llm.inference(
        messages,
        model=model,
        provider=provider,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        trace_id=trace_id,
        invocation_id=invocation_id,
        extra_headers=extra_headers,
    )


class _ExtractedResponse(TypedDict):
    """Fields pulled from a provider SDK response, ready for ``log_llm_call``."""

    response_content: str | None
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[_JsonObject] | None
    finish_reason: str


async def log_llm_call(
    input_messages: Sequence[_JsonObject | _LLMMessage],
    response_content: str | None = None,
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    tool_calls: list[_JsonObject] | None = None,
    finish_reason: str = "stop",
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an LLM call made to an external service for trace correlation.

    You do not need this function when using Dispatch's built-in LLM client:
    :func:`chat` and :func:`inference` log calls automatically.

    Use this when calling LLM providers directly through their SDKs so those
    calls appear in Dispatch traces alongside other agent activity.
    """
    api_base_url = _get_api_base_url()

    message_dicts: list[_JsonObject] = []
    for msg in input_messages:
        if isinstance(msg, _LLMMessage):
            message_dicts.append(msg.model_dump(exclude_none=True))
        else:
            message_dicts.append(msg)

    if trace_id is None:
        trace_id = _get_current_trace_id()
    if invocation_id is None:
        invocation_id = _get_current_invocation_id()

    payload: dict[str, Any] = {
        "input_messages": message_dicts,
        "response_content": response_content,
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
    }

    if tool_calls is not None:
        payload["tool_calls"] = tool_calls
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if trace_id is not None:
        payload["trace_id"] = trace_id
    if invocation_id is not None:
        payload["invocation_id"] = invocation_id

    agent_name = _config.agent_name
    if agent_name:
        payload["agent_name"] = agent_name

    url = f"{api_base_url}/llm/log"
    auth_headers = _get_auth_headers()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=payload,
            headers=auth_headers,
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

    return data["llm_call_id"]


def _extract_openai_response(response: Any) -> _ExtractedResponse:
    """Extract log_llm_call() fields from an OpenAI ChatCompletion response."""
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    content = message.content if message else None

    tool_calls = None
    if message and message.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]

    return {
        "response_content": content,
        "model": response.model,
        "provider": "openai",
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
        "tool_calls": tool_calls,
        "finish_reason": choice.finish_reason if choice else "stop",
    }


def _extract_anthropic_response(response: Any) -> _ExtractedResponse:
    """Extract log_llm_call() fields from an Anthropic Message response."""
    content = None
    tool_calls = None

    if response.content:
        text_blocks = []
        tool_use_blocks = []

        for block in response.content:
            if hasattr(block, "text"):
                text_blocks.append(block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_use_blocks.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": (
                                block.input
                                if isinstance(block.input, str)
                                else str(block.input)
                            ),
                        },
                    }
                )

        if text_blocks:
            content = "\n".join(text_blocks)
        if tool_use_blocks:
            tool_calls = tool_use_blocks

    finish_reason_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    finish_reason = finish_reason_map.get(response.stop_reason, response.stop_reason)

    return {
        "response_content": content,
        "model": response.model,
        "provider": "anthropic",
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }


def _is_openai_response(response: Any) -> bool:
    """Check if response is an OpenAI ChatCompletion."""
    return (
        hasattr(response, "choices")
        and hasattr(response, "usage")
        and hasattr(response, "model")
        and hasattr(response.usage, "prompt_tokens")
    )


def _is_anthropic_response(response: Any) -> bool:
    """Check if response is an Anthropic Message."""
    return (
        hasattr(response, "content")
        and hasattr(response, "usage")
        and hasattr(response, "stop_reason")
        and hasattr(response.usage, "input_tokens")
    )


async def log_openai_response(
    input_messages: Sequence[_JsonObject],
    response: Any,
    *,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an OpenAI ChatCompletion response by auto-extracting fields."""
    extracted = _extract_openai_response(response)
    return await log_llm_call(
        input_messages=input_messages,
        response_content=extracted["response_content"],
        model=extracted["model"],
        provider=extracted["provider"],
        input_tokens=extracted["input_tokens"],
        output_tokens=extracted["output_tokens"],
        tool_calls=extracted["tool_calls"],
        finish_reason=extracted["finish_reason"],
        latency_ms=latency_ms,
        trace_id=trace_id,
        invocation_id=invocation_id,
    )


async def log_anthropic_response(
    input_messages: Sequence[_JsonObject],
    response: Any,
    *,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an Anthropic Message response by auto-extracting fields."""
    extracted = _extract_anthropic_response(response)
    return await log_llm_call(
        input_messages=input_messages,
        response_content=extracted["response_content"],
        model=extracted["model"],
        provider=extracted["provider"],
        input_tokens=extracted["input_tokens"],
        output_tokens=extracted["output_tokens"],
        tool_calls=extracted["tool_calls"],
        finish_reason=extracted["finish_reason"],
        latency_ms=latency_ms,
        trace_id=trace_id,
        invocation_id=invocation_id,
    )


async def log_response(
    input_messages: Sequence[_JsonObject],
    response: Any,
    *,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an LLM response for trace correlation by auto-detecting provider.

    Raises:
        ValueError: If the response type is not recognized.
    """
    if _is_openai_response(response):
        return await log_openai_response(
            input_messages,
            response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            invocation_id=invocation_id,
        )
    if _is_anthropic_response(response):
        return await log_anthropic_response(
            input_messages,
            response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            invocation_id=invocation_id,
        )
    raise ValueError(
        "Unrecognized response type. Use log_openai_response(), "
        "log_anthropic_response(), or log_llm_call() with manual fields."
    )

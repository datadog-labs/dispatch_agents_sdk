"""LLM inference client for Dispatch agents.

Provides easy access to LLM inference via the Dispatch proxy with automatic
trace correlation. LLM calls made inside handler functions (@fn() or @on())
are automatically correlated with the invocation trace.

IMPORTANT: LLM calls should be made inside handler functions, not at module level.
Calls made outside handlers won't be associated with any trace.

Example:
    from dispatch_agents import fn, llm

    @fn()
    async def my_handler(payload):
        # Simple chat (one-off message)
        response = await llm.chat("What is 2+2?")
        print(response.content)  # "4"

        # With system prompt
        response = await llm.chat(
            "Summarize this document",
            system="You are a helpful assistant that summarizes text concisely."
        )

        # Full conversation with message history
        response = await llm.inference([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's the weather?"}
        ])
        return response.content

    # With structured output (JSON mode)
    from pydantic import BaseModel

    class Analysis(BaseModel):
        sentiment: str
        confidence: float

    @fn()
    async def analyze_sentiment(payload):
        response = await llm.chat(
            f"Analyze: {payload.text}",
            response_format=Analysis
        )
        return response.parse_json(Analysis)

    # With tool calling
    @fn()
    async def agent_with_tools(payload):
        tools = [{"type": "function", "function": {"name": "get_weather", ...}}]
        response = await llm.inference([{"role": "user", "content": payload.query}], tools=tools)
        if response.tool_calls:
            for call in response.tool_calls:
                print(f"Call {call.function.name} with {call.function.arguments}")
        return response.content
"""

import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, overload

import httpx
from pydantic import BaseModel

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

from .events import (
    _get_api_base_url,
    _get_auth_headers,
    get_current_invocation_id,
    get_current_trace_id,
)

# ContextVar for per-request extra headers to forward to LLM providers.
# Used by the extra_headers() context manager — async-safe so concurrent
# handler invocations each get their own copy.
_extra_llm_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "extra_llm_headers", default=None
)


@contextmanager
def extra_headers(headers: dict[str, str]) -> Generator[None, None, None]:
    """Context manager to attach extra headers to LLM provider requests.

    Headers set here are forwarded through the Dispatch proxy to the
    underlying LLM provider (e.g., an internal OpenAI-compatible gateway).
    Nested contexts merge with outer ones; inner keys override outer keys.

    Example:
        from dispatch_agents import extra_headers

        @fn()
        async def my_handler(payload):
            with extra_headers({"X-Dataset-Id": "team-ml"}):
                response = await llm.chat("Hello!")  # X-Dataset-Id sent to provider
    """
    current = _extra_llm_headers.get() or {}
    merged = {**current, **headers}
    token = _extra_llm_headers.set(merged)
    try:
        yield
    finally:
        _extra_llm_headers.reset(token)


def get_extra_llm_headers() -> dict[str, str]:
    """Return the current extra LLM headers (empty dict if none set)."""
    return _extra_llm_headers.get() or {}


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""

    role: str  # system, user, assistant, tool
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None


class LLMFunctionCall(BaseModel):
    """A function call within an LLM tool call."""

    name: str
    # "arguments" is a JSON-encoded string per the OpenAI chat completions API
    # (e.g. '{"location": "NYC"}'), not a collection. The singular concept is
    # "the arguments blob"; the plural name mirrors the upstream API field name.
    arguments: str


class LLMToolCall(BaseModel):
    """A tool call from the LLM response."""

    id: str
    type: str = "function"
    function: LLMFunctionCall


class LLMResponse(BaseModel):
    """Response from LLM inference."""

    llm_call_id: str
    content: str | None
    tool_calls: list[LLMToolCall] | None
    finish_reason: str
    model: str
    provider: str
    variant_name: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int

    def __str__(self) -> str:
        """Return the content for easy string conversion."""
        return self.content or ""

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    # @overload lets type checkers narrow the return type based on whether a
    # model class is passed:
    #   response.parse_json(MyModel)  -> MyModel
    #   response.parse_json()         -> dict[str, Any]
    #
    # We use overloads instead of making LLMResponse generic (e.g.
    # LLMResponse[T]) because LLMResponse is constructed inside inference()
    # from raw HTTP data — the target model type is only known later at parse
    # time, not at response construction time.  Pydantic generics require the
    # type parameter to be bound at class instantiation, which doesn't fit
    # this deferred-parsing pattern.
    @overload
    def parse_json(self, model: type[BaseModelT]) -> BaseModelT: ...

    @overload
    def parse_json(self, model: None = None) -> dict[str, Any]: ...

    def parse_json(
        self, model: type[BaseModel] | None = None
    ) -> dict[str, Any] | BaseModel:
        """Parse the response content as JSON.

        Args:
            model: Optional Pydantic model to validate against

        Returns:
            Parsed JSON as dict, or validated Pydantic model if provided

        Raises:
            ValueError: If content is not valid JSON
        """
        import json

        if not self.content:
            raise ValueError("Response has no content to parse")

        data = json.loads(self.content)
        if model is not None:
            return model.model_validate(data)
        return data


class LLMClient:
    """Client for LLM inference via Dispatch proxy.

    Automatically propagates trace context for correlation with agent invocations.

    Example:
        from dispatch_agents import llm

        # Simple one-liner
        response = await llm.chat("What is Python?")

        # With system prompt
        response = await llm.chat(
            "Explain quantum computing",
            system="You explain complex topics simply."
        )

        # Full conversation history
        response = await llm.inference([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ])

        # With structured output
        response = await llm.chat("List 3 colors", response_format={"type": "json_object"})
        colors = response.parse_json()
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize LLM client with optional defaults.

        Args:
            model: Default model to use (e.g., "gpt-4o", "claude-3-5-sonnet")
            provider: Default provider (e.g., "openai", "anthropic")
            temperature: Default sampling temperature (0-2)
            max_tokens: Default maximum tokens in response
        """
        self._api_base_url: str | None = None
        self._default_model = model
        self._default_provider = provider
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens

    def _ensure_api_base_url(self) -> str:
        """Lazily initialize API base URL when first needed."""
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
        response_format: dict[str, Any] | type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Simple chat interface for one-off messages.

        This is the easiest way to call an LLM - just pass a string!

        Args:
            message: The user message to send
            system: Optional system prompt
            model: Model to use (uses client default or org default if not specified)
            provider: Provider to use (uses client default or org default if not specified)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Request structured output. Can be:
                - {"type": "json_object"} for JSON mode
                - A Pydantic model class for schema-guided generation

        Returns:
            LLMResponse with content, usage metrics, and cost

        Example:
            # Basic
            response = await llm.chat("What is 2+2?")
            print(response.content)

            # With system prompt
            response = await llm.chat(
                "Summarize this text",
                system="You summarize text in exactly 3 bullet points."
            )

            # Structured output with Pydantic model
            class Colors(BaseModel):
                colors: list[str]

            response = await llm.chat(
                "List 3 primary colors",
                response_format=Colors
            )
            result = response.parse_json(Colors)
            print(result.colors)  # ['red', 'blue', 'yellow']
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        # Handle response_format - convert Pydantic model to JSON schema
        format_dict: dict[str, Any] | None = None
        if response_format is not None:
            if isinstance(response_format, dict):
                format_dict = response_format
            elif isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                # Convert Pydantic model to JSON schema
                format_dict = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema(),
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
        messages: Sequence[dict[str, Any] | LLMMessage],
        *,
        model: str | None = None,
        provider: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        trace_id: str | None = None,
        invocation_id: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Execute LLM inference via Dispatch proxy.

        Automatically includes trace context from the current execution for
        correlation with agent invocations in observability tools.

        Args:
            messages: Conversation messages (list of dicts with role/content)
            model: Model to use (e.g., "gpt-4o", "claude-sonnet-4-5").
                   If omitted, falls back to the provider's configured default_model.
            provider: Provider to route the request to (e.g., "openai", "anthropic").
                      If omitted, falls back to the org's ``default_provider``.
                      If no default is configured, the request will fail with an error.
                      **Tip:** always pass ``provider=`` explicitly when you pass
                      ``model=`` to avoid accidentally sending a model name to the
                      wrong provider.
            tools: Tool definitions for function calling
            temperature: Sampling temperature (0-2). Uses client default if not specified.
            max_tokens: Maximum tokens in response. Uses client default if not specified.
            response_format: Request structured output format (e.g., {"type": "json_object"})
            trace_id: Override trace ID (auto-detected from handler context if not provided)
            invocation_id: Override invocation ID (auto-detected from handler context if not provided).
                          This links the LLM call to its parent invocation in the trace tree.

        Returns:
            LLMResponse with content, usage metrics, and cost

        Raises:
            httpx.HTTPStatusError: If the request fails
            RuntimeError: If DISPATCH_NAMESPACE is not set

        Example:
            response = await llm_client.inference([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ])
            print(f"Answer: {response.content}")
            print(f"Cost: ${response.cost_usd:.4f}")
        """
        api_base_url = self._ensure_api_base_url()

        # Convert LLMMessage objects to dicts
        message_dicts = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                message_dicts.append(msg.model_dump(exclude_none=True))
            else:
                message_dicts.append(msg)

        # Auto-detect context from current execution if not provided
        # This enables automatic trace correlation when called from within a handler
        if trace_id is None:
            trace_id = get_current_trace_id()
        if invocation_id is None:
            invocation_id = get_current_invocation_id()

        # Apply client defaults
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

        # Build request payload
        payload: dict[str, Any] = {
            "messages": message_dicts,
        }

        # Only include temperature if we have a value
        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        if effective_model is not None:
            payload["model"] = effective_model
        if effective_provider is not None:
            payload["provider"] = effective_provider
        if tools is not None:
            payload["tools"] = tools
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if invocation_id is not None:
            payload["invocation_id"] = invocation_id

        # Include agent name for cost tracking and budget enforcement
        agent_name = os.environ.get("DISPATCH_AGENT_NAME")
        if agent_name:
            payload["agent_name"] = agent_name

        # Merge extra headers: ContextVar first, then explicit param overrides
        merged_headers = {**get_extra_llm_headers()}
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
                timeout=600.0,  # 10min — matches ALB idle timeout for long-context LLM calls
            )
            response.raise_for_status()
            data = response.json()

        # Parse tool calls if present
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [LLMToolCall(**tc) for tc in data["tool_calls"]]

        return LLMResponse(
            llm_call_id=data["llm_call_id"],
            content=data.get("content"),
            tool_calls=tool_calls,
            finish_reason=data["finish_reason"],
            model=data["model"],
            provider=data["provider"],
            variant_name=data.get("variant_name"),
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost_usd=data["cost_usd"],
            latency_ms=data["latency_ms"],
        )


# Module-level singleton for convenient access
llm = LLMClient()


# Convenience functions for direct usage
async def chat(
    message: str,
    *,
    system: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
) -> LLMResponse:
    """Simple chat interface for one-off messages.

    This is a convenience function that uses the module-level singleton.
    See LLMClient.chat() for full documentation.

    Example:
        from dispatch_agents.llm import chat

        response = await chat("What is 2+2?")
        print(response.content)

        # With system prompt
        response = await chat(
            "Explain quantum computing",
            system="You explain complex topics simply."
        )
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
    messages: Sequence[dict[str, Any] | LLMMessage],
    *,
    model: str | None = None,
    provider: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> LLMResponse:
    """Execute LLM inference via Dispatch proxy.

    This is a convenience function that uses the module-level singleton.
    See LLMClient.inference() for full documentation.

    Example:
        from dispatch_agents.llm import inference

        response = await inference([
            {"role": "user", "content": "Hello!"}
        ])
        print(response.content)
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


async def log_llm_call(
    input_messages: Sequence[dict[str, Any] | LLMMessage],
    response_content: str | None = None,
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an LLM call made to an external service for trace correlation.

    IMPORTANT: You do NOT need this function if you use Dispatch's built-in LLM client!
    The llm.chat() and llm.inference() functions automatically log calls for you.

    This function is ONLY needed when you call LLM providers directly using their
    SDKs (OpenAI, Anthropic, etc.) instead of Dispatch's llm.chat()/inference() proxy.
    It enables those external calls to appear in Dispatch traces alongside other
    agent activity.

    When to use this function:
    - You're using the OpenAI SDK directly for streaming or advanced features
    - You have existing code using provider SDKs that you don't want to migrate
    - You need features not yet supported by Dispatch's LLM client

    When NOT to use this function:
    - You're using llm.chat() or llm.inference() - they log automatically!

    Args:
        input_messages: The conversation messages sent to the LLM (full context, not deltas)
        response_content: The text content of the LLM's response
        model: Model used (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        provider: Provider name (e.g., "openai", "anthropic")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        tool_calls: Tool/function calls returned by the LLM (optional)
        finish_reason: Reason the generation stopped (default: "stop")
        latency_ms: Time taken in milliseconds (optional)
        trace_id: Override trace ID (auto-detected from handler context if not provided)
        invocation_id: Override invocation ID (auto-detected from handler context)

    Returns:
        The llm_call_id assigned to this logged call

    Example:
        # Using OpenAI client directly (only do this if you need features
        # not available in llm.chat(), otherwise just use llm.chat()!)
        from openai import AsyncOpenAI
        from dispatch_agents import llm

        client = AsyncOpenAI()
        messages = [{"role": "user", "content": "Hello!"}]

        # Make the call directly to OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        # Log it to Dispatch for trace visibility
        await llm.log_llm_call(
            input_messages=messages,
            response_content=response.choices[0].message.content,
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            finish_reason=response.choices[0].finish_reason,
        )
    """
    api_base_url = _get_api_base_url()

    # Convert LLMMessage objects to dicts
    message_dicts = []
    for msg in input_messages:
        if isinstance(msg, LLMMessage):
            message_dicts.append(msg.model_dump(exclude_none=True))
        else:
            message_dicts.append(msg)

    # Auto-detect context from current execution if not provided
    if trace_id is None:
        trace_id = get_current_trace_id()
    if invocation_id is None:
        invocation_id = get_current_invocation_id()

    # Build request payload
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

    # Include agent name for cost tracking
    agent_name = os.environ.get("DISPATCH_AGENT_NAME")
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


# =============================================================================
# Ergonomic helpers for popular SDKs
# =============================================================================
# These functions auto-extract fields from SDK response objects so users
# don't have to manually pull out tokens, content, etc.


def _extract_openai_response(response: Any) -> dict[str, Any]:
    """Extract fields from an OpenAI ChatCompletion response.

    Works with both sync and async OpenAI SDK responses.

    Args:
        response: OpenAI ChatCompletion object

    Returns:
        Dict with extracted fields for log_llm_call()
    """
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None

    # Extract content
    content = message.content if message else None

    # Extract tool calls (OpenAI format)
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


def _extract_anthropic_response(response: Any) -> dict[str, Any]:
    """Extract fields from an Anthropic Message response.

    Args:
        response: Anthropic Message object

    Returns:
        Dict with extracted fields for log_llm_call()
    """
    # Extract text content (Anthropic uses content blocks)
    content = None
    tool_calls = None

    if response.content:
        text_blocks = []
        tool_use_blocks = []

        for block in response.content:
            # Duck type check for text block
            if hasattr(block, "text"):
                text_blocks.append(block.text)
            # Duck type check for tool_use block
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

    # Map Anthropic stop_reason to standard finish_reason
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
    input_messages: Sequence[dict[str, Any]],
    response: Any,
    *,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an OpenAI ChatCompletion response for trace correlation.

    This is a convenience wrapper around log_llm_call() that automatically
    extracts fields from the OpenAI response object.

    IMPORTANT: You do NOT need this if you use llm.chat() - it logs automatically!

    Args:
        input_messages: The messages array you sent to OpenAI
        response: The ChatCompletion response from OpenAI
        latency_ms: Time taken in milliseconds (optional)
        trace_id: Override trace ID (auto-detected from handler context)
        invocation_id: Override invocation ID (auto-detected from handler context)

    Returns:
        The llm_call_id assigned to this logged call

    Example:
        from openai import AsyncOpenAI
        from dispatch_agents import llm

        client = AsyncOpenAI()
        messages = [{"role": "user", "content": "Hello!"}]

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        # One line to log - no manual field extraction!
        await llm.log_openai_response(messages, response)
    """
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
    input_messages: Sequence[dict[str, Any]],
    response: Any,
    *,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an Anthropic Message response for trace correlation.

    This is a convenience wrapper around log_llm_call() that automatically
    extracts fields from the Anthropic response object.

    IMPORTANT: You do NOT need this if you use llm.chat() - it logs automatically!

    Args:
        input_messages: The messages array you sent to Anthropic
        response: The Message response from Anthropic
        latency_ms: Time taken in milliseconds (optional)
        trace_id: Override trace ID (auto-detected from handler context)
        invocation_id: Override invocation ID (auto-detected from handler context)

    Returns:
        The llm_call_id assigned to this logged call

    Example:
        import anthropic
        from dispatch_agents import llm

        client = anthropic.AsyncAnthropic()
        messages = [{"role": "user", "content": "Hello!"}]

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages,
        )

        # One line to log - no manual field extraction!
        await llm.log_anthropic_response(messages, response)
    """
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
    input_messages: Sequence[dict[str, Any]],
    response: Any,
    *,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
) -> str:
    """Log an LLM response for trace correlation (auto-detects provider).

    This function automatically detects whether the response is from OpenAI
    or Anthropic and extracts the appropriate fields.

    IMPORTANT: You do NOT need this if you use llm.chat() - it logs automatically!

    Args:
        input_messages: The messages array you sent to the LLM
        response: The response object from OpenAI or Anthropic
        latency_ms: Time taken in milliseconds (optional)
        trace_id: Override trace ID (auto-detected from handler context)
        invocation_id: Override invocation ID (auto-detected from handler context)

    Returns:
        The llm_call_id assigned to this logged call

    Raises:
        ValueError: If the response type is not recognized

    Example:
        from dispatch_agents import llm

        # Works with OpenAI
        response = await openai_client.chat.completions.create(...)
        await llm.log_response(messages, response)

        # Works with Anthropic
        response = await anthropic_client.messages.create(...)
        await llm.log_response(messages, response)
    """
    if _is_openai_response(response):
        return await log_openai_response(
            input_messages,
            response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            invocation_id=invocation_id,
        )
    elif _is_anthropic_response(response):
        return await log_anthropic_response(
            input_messages,
            response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            invocation_id=invocation_id,
        )
    else:
        raise ValueError(
            "Unrecognized response type. Use log_openai_response(), "
            "log_anthropic_response(), or log_llm_call() with manual fields."
        )

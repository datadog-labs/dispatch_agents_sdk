"""LLM Gateway Tester — exercises the full LLM surface area.

Tests four layers:
1. Native OpenAI SDK — chat, streaming, embeddings, models list, responses, tool calls
2. Native Anthropic SDK — messages, streaming, tool calls
3. Dispatch LLM SDK — chat, inference, tools, extra headers, structured output, multi-turn
4. MCP Integration — list tools, call a tool via get_mcp_client

Each @fn() returns a TestResult with status, response preview, error, and
elapsed time so you can quickly verify what works.
"""

import json
import time
from datetime import UTC, datetime
from typing import Any

from dispatch_agents import BasePayload, fn

# =============================================================================
# Shared types and helpers
# =============================================================================


class TestResult(BasePayload):
    """Common response from all test functions."""

    status: str  # "ok" or "error"
    response_preview: str | None = None
    error: str | None = None
    elapsed_ms: int = 0
    extra: dict[str, Any] | None = None


def _ok(preview: str, elapsed_ms: int, **extra: Any) -> TestResult:
    return TestResult(
        status="ok",
        response_preview=preview[:2000] if preview else None,
        elapsed_ms=elapsed_ms,
        extra=extra or None,
    )


def _err(error: str, elapsed_ms: int) -> TestResult:
    return TestResult(status="error", error=error, elapsed_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Real tool implementations (used by tool-call tests instead of mocks)
# ---------------------------------------------------------------------------


def _register_tool(name: str, description: str, parameters: dict) -> dict:
    """Return an OpenAI-format tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _execute_tool(name: str, arguments: dict) -> str:
    """Execute a registered tool and return its string result."""
    if name == "get_current_time":
        return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    elif name == "calculate":
        expr = arguments.get("expression", "")
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expr):
            return f"Error: invalid characters in expression: {expr}"
        try:
            return str(eval(expr))  # noqa: S307 — restricted charset
        except Exception as e:
            return f"Error: {e}"
    elif name == "get_weather":
        location = arguments.get("location", "unknown")
        return json.dumps({"location": location, "temp_f": 72, "condition": "sunny"})
    return f"Unknown tool: {name}"


TOOLS_OPENAI = [
    _register_tool(
        "get_current_time",
        "Get the current UTC time.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _register_tool(
        "calculate",
        "Evaluate a simple math expression.",
        {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression like '2 + 2' or '3.14 * 10'",
                }
            },
            "required": ["expression"],
        },
    ),
    _register_tool(
        "get_weather",
        "Get the current weather for a location.",
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    ),
]

# Anthropic uses a different tool schema format
TOOLS_ANTHROPIC = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS_OPENAI
]


# =============================================================================
# 1. Native OpenAI SDK tests
# =============================================================================


class OpenAIChatRequest(BasePayload):
    message: str = "Say hello in exactly 5 words."
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024


@fn()
async def test_openai_chat(payload: OpenAIChatRequest) -> TestResult:
    """Test OpenAI chat completions (non-streaming)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=payload.model,
            messages=[{"role": "user", "content": payload.message}],
            max_completion_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        content = resp.choices[0].message.content or ""
        return _ok(
            content,
            elapsed,
            model=resp.model,
            finish_reason=resp.choices[0].finish_reason,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class StreamingRequest(BasePayload):
    message: str = "Count from 1 to 20, one number per line."
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024


@fn()
async def test_openai_streaming(payload: StreamingRequest) -> TestResult:
    """Test OpenAI streaming chat completions."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        stream = await client.chat.completions.create(
            model=payload.model,
            messages=[{"role": "user", "content": payload.message}],
            stream=True,
            max_completion_tokens=payload.max_tokens,
        )
        chunks: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        elapsed = int((time.monotonic() - start) * 1000)
        full_text = "".join(chunks)
        return _ok(
            full_text,
            elapsed,
            chunk_count=len(chunks),
            note="Streaming worked! (may be buffered by proxy)",
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(f"Streaming error: {e}", elapsed)


class OpenAIToolCallRequest(BasePayload):
    query: str = "What is 123 * 456? Use the calculate tool."
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024


@fn()
async def test_openai_tool_call(payload: OpenAIToolCallRequest) -> TestResult:
    """Test OpenAI SDK with real tool execution (non-streaming).

    Sends tools to the LLM, executes the returned tool call against a real
    Python function, feeds the result back, and gets the final answer.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": payload.query},
        ]
        resp = await client.chat.completions.create(
            model=payload.model,
            messages=messages,
            tools=TOOLS_OPENAI,
            max_completion_tokens=payload.max_tokens,
        )
        choice = resp.choices[0]

        if not choice.message.tool_calls:
            elapsed = int((time.monotonic() - start) * 1000)
            return _ok(
                choice.message.content or "(no tool call returned)",
                elapsed,
                tool_calls=None,
            )

        # Execute real tool calls
        messages.append(choice.message.model_dump())
        tool_results = []
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, args)
            tool_results.append(
                {"tool": tc.function.name, "args": args, "result": result}
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Get final answer
        final = await client.chat.completions.create(
            model=payload.model,
            messages=messages,
            max_completion_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            final.choices[0].message.content or "",
            elapsed,
            tool_results=tool_results,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


@fn()
async def test_openai_tool_call_streaming(
    payload: OpenAIToolCallRequest,
) -> TestResult:
    """Test OpenAI SDK tool calling with streaming enabled.

    Same as test_openai_tool_call but uses stream=True to verify that
    tool_calls are correctly assembled from streamed deltas.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": payload.query},
        ]
        stream = await client.chat.completions.create(
            model=payload.model,
            messages=messages,
            tools=TOOLS_OPENAI,
            stream=True,
            max_completion_tokens=payload.max_tokens,
        )

        # Assemble streamed response
        content_chunks: list[str] = []
        tool_calls_by_idx: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if delta.content:
                content_chunks.append(delta.content)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_idx:
                        tool_calls_by_idx[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_by_idx[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

        if not tool_calls_by_idx:
            elapsed = int((time.monotonic() - start) * 1000)
            return _ok(
                "".join(content_chunks) or "(no tool call in stream)",
                elapsed,
                tool_calls=None,
            )

        # Execute tool calls
        assembled_calls = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": assembled_calls}
        )
        tool_results = []
        for tc in assembled_calls:
            args = json.loads(tc["function"]["arguments"])
            result = _execute_tool(tc["function"]["name"], args)
            tool_results.append(
                {"tool": tc["function"]["name"], "args": args, "result": result}
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

        # Final answer (non-streaming)
        final = await client.chat.completions.create(
            model=payload.model,
            messages=messages,
            max_completion_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            final.choices[0].message.content or "",
            elapsed,
            tool_results=tool_results,
            streamed_tool_call=True,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class EmbeddingsRequest(BasePayload):
    text: str = "The quick brown fox jumps over the lazy dog"
    model: str = "text-embedding-3-small"


@fn()
async def test_openai_embeddings(payload: EmbeddingsRequest) -> TestResult:
    """Test OpenAI embeddings (catch-all passthrough path)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        resp = await client.embeddings.create(model=payload.model, input=payload.text)
        elapsed = int((time.monotonic() - start) * 1000)
        embedding = resp.data[0].embedding
        return _ok(
            f"Got {len(embedding)}-dim embedding",
            elapsed,
            dimensions=len(embedding),
            model=resp.model,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class ModelsListRequest(BasePayload):
    pass


@fn()
async def test_openai_models_list(payload: ModelsListRequest) -> TestResult:
    """Test OpenAI models list (GET request via catch-all passthrough)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        models = await client.models.list()
        elapsed = int((time.monotonic() - start) * 1000)
        model_ids = [m.id for m in models.data[:5]]
        return _ok(
            f"Found {len(models.data)} models: {', '.join(model_ids)}...",
            elapsed,
            model_count=len(models.data),
            sample_models=model_ids,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class ResponsesRequest(BasePayload):
    message: str = "What is 2+2? Answer with just the number."
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024


@fn()
async def test_openai_responses(payload: ResponsesRequest) -> TestResult:
    """Test OpenAI Responses API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.monotonic()
    try:
        resp = await client.responses.create(
            model=payload.model,
            input=payload.message,
            max_output_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        text = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        text += block.text
        return _ok(text or str(resp.output), elapsed, model=resp.model)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


# =============================================================================
# 2. Native Anthropic SDK tests
# =============================================================================


class AnthropicChatRequest(BasePayload):
    message: str = "Say hello in exactly 5 words."
    model: str = "claude-3-5-haiku-20241022"
    max_tokens: int = 1024


@fn()
async def test_anthropic_chat(payload: AnthropicChatRequest) -> TestResult:
    """Test Anthropic messages (non-streaming)."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    start = time.monotonic()
    try:
        resp = await client.messages.create(
            model=payload.model,
            max_tokens=payload.max_tokens,
            messages=[{"role": "user", "content": payload.message}],
        )
        elapsed = int((time.monotonic() - start) * 1000)
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
        return _ok(
            text,
            elapsed,
            model=resp.model,
            stop_reason=resp.stop_reason,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


@fn()
async def test_anthropic_streaming(payload: AnthropicChatRequest) -> TestResult:
    """Test Anthropic streaming messages."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    start = time.monotonic()
    try:
        chunks: list[str] = []
        async with client.messages.stream(
            model=payload.model,
            max_tokens=payload.max_tokens,
            messages=[{"role": "user", "content": payload.message}],
        ) as stream:
            async for text in stream.text_stream:
                chunks.append(text)
        elapsed = int((time.monotonic() - start) * 1000)
        full_text = "".join(chunks)
        return _ok(
            full_text,
            elapsed,
            chunk_count=len(chunks),
            note="Streaming worked! (may be buffered by proxy)",
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(f"Streaming error: {e}", elapsed)


class AnthropicToolCallRequest(BasePayload):
    query: str = "What is 123 * 456? Use the calculate tool."
    model: str = "claude-3-5-haiku-20241022"
    max_tokens: int = 1024


@fn()
async def test_anthropic_tool_call(payload: AnthropicToolCallRequest) -> TestResult:
    """Test Anthropic SDK with real tool execution.

    Uses Anthropic's native tool_use format. Executes the tool call against
    a real Python function and feeds the result back.
    """
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    start = time.monotonic()
    try:
        resp = await client.messages.create(
            model=payload.model,
            max_tokens=payload.max_tokens,
            messages=[{"role": "user", "content": payload.query}],
            tools=TOOLS_ANTHROPIC,
        )

        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        if not tool_use_blocks:
            elapsed = int((time.monotonic() - start) * 1000)
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return _ok(text or "(no tool call returned)", elapsed, tool_calls=None)

        # Execute real tool calls
        tool_results_content: list[dict[str, Any]] = []
        tool_results_log: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            result = _execute_tool(block.name, block.input)
            tool_results_log.append(
                {"tool": block.name, "args": block.input, "result": result}
            )
            tool_results_content.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )

        # Get final answer
        final = await client.messages.create(
            model=payload.model,
            max_tokens=payload.max_tokens,
            messages=[
                {"role": "user", "content": payload.query},
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": tool_results_content},
            ],
            tools=TOOLS_ANTHROPIC,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        text = "".join(b.text for b in final.content if hasattr(b, "text"))
        return _ok(text, elapsed, tool_results=tool_results_log)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


# =============================================================================
# 3. Dispatch LLM SDK tests
# =============================================================================


class DispatchChatRequest(BasePayload):
    message: str = "Say hello in exactly 5 words."
    model: str | None = None
    provider: str | None = None
    max_tokens: int = 1024


@fn()
async def test_dispatch_chat(payload: DispatchChatRequest) -> TestResult:
    """Test Dispatch llm.chat() — the high-level convenience API."""
    from dispatch_agents import llm

    start = time.monotonic()
    try:
        resp = await llm.chat(
            payload.message,
            system="You are a helpful assistant. Be brief.",
            model=payload.model,
            provider=payload.provider,
            max_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            resp.content or "",
            elapsed,
            model=resp.model,
            provider=resp.provider,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            llm_call_id=resp.llm_call_id,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchAnthropicRequest(BasePayload):
    message: str = "Say hello in exactly 5 words."
    model: str = "claude-3-5-haiku-20241022"
    max_tokens: int = 1024


@fn()
async def test_dispatch_anthropic(payload: DispatchAnthropicRequest) -> TestResult:
    """Test Dispatch llm.chat() routed to Anthropic."""
    from dispatch_agents import llm

    start = time.monotonic()
    try:
        resp = await llm.chat(
            payload.message,
            model=payload.model,
            provider="anthropic",
            max_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            resp.content or "",
            elapsed,
            model=resp.model,
            provider=resp.provider,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchToolCallRequest(BasePayload):
    query: str = "What is 123 * 456? Use the calculate tool."
    max_tokens: int = 1024


@fn()
async def test_dispatch_tool_call(payload: DispatchToolCallRequest) -> TestResult:
    """Test Dispatch llm.inference() with real tool execution.

    Defines tools, lets the LLM call them, executes the real Python function,
    feeds the result back, and gets the final answer.
    """
    from dispatch_agents import llm

    start = time.monotonic()
    try:
        first = await llm.inference(
            [{"role": "user", "content": payload.query}],
            model="gpt-4o-mini",
            provider="openai",
            tools=TOOLS_OPENAI,
            temperature=0,
            max_tokens=payload.max_tokens,
        )

        if not first.tool_calls:
            elapsed = int((time.monotonic() - start) * 1000)
            return _ok(
                first.content or "(no tool call returned)", elapsed, tool_calls=None
            )

        tool_call_data = [tc.model_dump() for tc in first.tool_calls]

        # Execute real tool calls
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": payload.query},
            {"role": "assistant", "content": None, "tool_calls": tool_call_data},
        ]
        tool_results = []
        for tc in first.tool_calls:
            args = (
                json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments
            )
            result = _execute_tool(tc.function.name, args)
            tool_results.append(
                {"tool": tc.function.name, "args": args, "result": result}
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final = await llm.inference(
            messages,
            model="gpt-4o-mini",
            provider="openai",
            max_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)

        return _ok(
            final.content or "",
            elapsed,
            tool_results=tool_results,
            total_cost_usd=first.cost_usd + final.cost_usd,
            total_tokens=first.total_tokens + final.total_tokens,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchExtraHeadersRequest(BasePayload):
    message: str = "Say hello."
    headers: dict[str, str] = {"X-Test-Header": "llm-gateway-tester"}
    max_tokens: int = 1024


@fn()
async def test_dispatch_extra_headers(
    payload: DispatchExtraHeadersRequest,
) -> TestResult:
    """Test Dispatch extra_headers() context manager.

    Verifies custom headers are forwarded through the proxy to the provider.
    """
    from dispatch_agents import llm
    from dispatch_agents.llm import extra_headers

    start = time.monotonic()
    try:
        with extra_headers(payload.headers):
            resp = await llm.chat(
                payload.message,
                model="gpt-4o-mini",
                provider="openai",
                max_tokens=payload.max_tokens,
            )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            resp.content or "",
            elapsed,
            headers_sent=payload.headers,
            model=resp.model,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchNestedHeadersRequest(BasePayload):
    message: str = "Say hello."
    max_tokens: int = 1024


@fn()
async def test_dispatch_nested_headers(
    payload: DispatchNestedHeadersRequest,
) -> TestResult:
    """Test nested extra_headers contexts — inner should override outer.

    Sets X-Team in outer scope, then overrides in inner scope.
    """
    from dispatch_agents import llm
    from dispatch_agents.llm import extra_headers

    start = time.monotonic()
    try:
        with extra_headers({"X-Team": "outer", "X-Org": "dd"}):
            with extra_headers({"X-Team": "inner"}):
                resp = await llm.chat(
                    payload.message,
                    model="gpt-4o-mini",
                    provider="openai",
                    max_tokens=payload.max_tokens,
                )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            resp.content or "",
            elapsed,
            expected_headers={"X-Team": "inner", "X-Org": "dd"},
            model=resp.model,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchStructuredRequest(BasePayload):
    message: str = "List 3 colors as a JSON array under key 'colors'."
    max_tokens: int = 1024


@fn()
async def test_dispatch_structured_output(
    payload: DispatchStructuredRequest,
) -> TestResult:
    """Test Dispatch llm.chat() with JSON structured output."""
    from dispatch_agents import llm

    start = time.monotonic()
    try:
        resp = await llm.chat(
            payload.message,
            model="gpt-4o-mini",
            provider="openai",
            response_format={"type": "json_object"},
            max_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        parsed = resp.parse_json()
        return _ok(resp.content or "", elapsed, parsed_json=parsed, model=resp.model)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchMultiTurnRequest(BasePayload):
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name?"},
    ]
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    max_tokens: int = 1024


@fn()
async def test_dispatch_multi_turn(payload: DispatchMultiTurnRequest) -> TestResult:
    """Test multi-turn conversation history via Dispatch inference()."""
    from dispatch_agents import llm

    start = time.monotonic()
    try:
        resp = await llm.inference(
            payload.messages,
            model=payload.model,
            provider=payload.provider,
            max_tokens=payload.max_tokens,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            resp.content or "",
            elapsed,
            model=resp.model,
            provider=resp.provider,
            turn_count=len(payload.messages),
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class DispatchCompareRequest(BasePayload):
    message: str = "Explain quantum computing in one sentence."
    providers: list[dict[str, str]] = [
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
    ]
    max_tokens: int = 1024


@fn()
async def test_dispatch_compare_providers(
    payload: DispatchCompareRequest,
) -> TestResult:
    """Send the same prompt to multiple providers and compare."""
    from dispatch_agents import llm

    start = time.monotonic()
    try:
        results = []
        total_cost = 0.0
        for spec in payload.providers:
            resp = await llm.chat(
                payload.message,
                model=spec.get("model"),
                provider=spec.get("provider"),
                max_tokens=payload.max_tokens,
            )
            total_cost += resp.cost_usd
            results.append(
                {
                    "provider": resp.provider,
                    "model": resp.model,
                    "response": (resp.content or "")[:500],
                    "tokens": resp.total_tokens,
                    "cost_usd": resp.cost_usd,
                    "latency_ms": resp.latency_ms,
                }
            )
        elapsed = int((time.monotonic() - start) * 1000)
        summary = " | ".join(
            f"{r['provider']}/{r['model']}: {r['latency_ms']}ms" for r in results
        )
        return _ok(summary, elapsed, results=results, total_cost_usd=total_cost)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


# =============================================================================
# 4. MCP Integration tests
# =============================================================================


class MCPListToolsRequest(BasePayload):
    server: str = "datadog"


@fn()
async def test_mcp_list_tools(payload: MCPListToolsRequest) -> TestResult:
    """List available tools from an MCP server.

    Connects to the specified MCP server (configured in dispatch.yaml) and
    lists all available tools. Useful for discovery before calling tools.
    """
    from dispatch_agents import get_mcp_client

    start = time.monotonic()
    try:
        async with get_mcp_client(payload.server) as client:
            tools_result = await client.list_tools()
            tool_names = [t.name for t in tools_result.tools]
        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            f"Found {len(tool_names)} tools: {', '.join(tool_names[:10])}...",
            elapsed,
            tool_count=len(tool_names),
            tool_names=tool_names[:20],
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class MCPCallToolRequest(BasePayload):
    server: str = "datadog"
    tool_name: str = "search_datadog_monitors"
    arguments: dict[str, Any] = {
        "telemetry": {"intent": "llm-gateway-tester: testing MCP tool call"},
    }


@fn()
async def test_mcp_call_tool(payload: MCPCallToolRequest) -> TestResult:
    """Call a specific tool on an MCP server.

    Directly invokes a tool on the MCP server with the given arguments.
    Tests the full MCP tool-call round trip including trace context injection.
    """
    from dispatch_agents import get_mcp_client

    start = time.monotonic()
    try:
        async with get_mcp_client(payload.server) as client:
            result = await client.call_tool(payload.tool_name, payload.arguments)
        elapsed = int((time.monotonic() - start) * 1000)

        content_preview = ""
        if result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    content_preview += item.text[:1000]

        return _ok(
            content_preview or "(empty response)",
            elapsed,
            tool=payload.tool_name,
            is_error=result.isError,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


# =============================================================================
# 5. Edge case tests
# =============================================================================


class RawEndpointRequest(BasePayload):
    path: str = "/v1/models"
    method: str = "GET"
    provider: str = "openai"


@fn()
async def test_raw_endpoint(payload: RawEndpointRequest) -> TestResult:
    """Test a raw HTTP call through the proxy to an arbitrary endpoint.

    Sends a raw httpx request through the proxy to verify the catch-all
    routing works for any /v1/* path.
    """
    import os

    import httpx

    proxy_url = os.environ.get("DISPATCH_LLM_PROXY_URL", "http://127.0.0.1:8780")
    url = f"{proxy_url}/{payload.provider}{payload.path}"

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            resp = await http_client.request(method=payload.method, url=url)
        elapsed = int((time.monotonic() - start) * 1000)

        if resp.status_code < 400:
            return _ok(
                resp.text[:500],
                elapsed,
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type"),
            )
        else:
            return _err(f"HTTP {resp.status_code}: {resp.text[:300]}", elapsed)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


# =============================================================================
# 6. Anthropic tool_use tests (web search, code execution)
# =============================================================================


class AnthropicWebSearchRequest(BasePayload):
    query: str = "What is the current population of Tokyo?"
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024


@fn()
async def test_anthropic_web_search(
    payload: AnthropicWebSearchRequest,
) -> TestResult:
    """Test Anthropic with the built-in web search tool.

    Sends a query that requires web search, verifies the model uses the
    server-side web_search tool and returns grounded results.
    """
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    start = time.monotonic()
    try:
        resp = await client.messages.create(
            model=payload.model,
            max_tokens=payload.max_tokens,
            messages=[{"role": "user", "content": payload.query}],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
            ],
        )
        elapsed = int((time.monotonic() - start) * 1000)

        text_parts = []
        tool_uses = []
        for block in resp.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append({"name": block.name, "id": block.id})
            elif block.type == "web_search_tool_result":
                tool_uses.append(
                    {
                        "type": "web_search_result",
                        "id": getattr(block, "tool_use_id", ""),
                    }
                )

        return _ok(
            "".join(text_parts) if text_parts else "(no text output)",
            elapsed,
            model=resp.model,
            stop_reason=resp.stop_reason,
            tool_uses=tool_uses,
            content_block_count=len(resp.content),
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)


class ClaudeCodeAgentRequest(BasePayload):
    task: str = "List the files in /tmp, check what network interfaces are available, and report the hostname."
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096


@fn()
async def test_claude_code_agent(payload: ClaudeCodeAgentRequest) -> TestResult:
    """Test Anthropic with bash tool for code execution.

    Uses Claude's built-in bash tool to run commands, exercising multi-turn
    tool_use → tool_result flow through the proxy. Verifies the proxy
    correctly handles Anthropic's tool_use/tool_result content blocks.
    """
    import subprocess

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    start = time.monotonic()
    try:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": payload.task},
        ]

        commands_run: list[dict[str, str]] = []
        max_turns = 5

        for _turn in range(max_turns):
            resp = await client.messages.create(
                model=payload.model,
                max_tokens=payload.max_tokens,
                system="You are a systems admin. Use the bash tool to complete tasks. Be concise.",
                messages=messages,
                tools=[
                    {
                        "type": "bash_20250124",
                        "name": "bash",
                    }
                ],
            )

            # If no tool use, we're done
            tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
            if not tool_use_blocks:
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                elapsed = int((time.monotonic() - start) * 1000)
                return _ok(
                    text,
                    elapsed,
                    model=resp.model,
                    commands_run=commands_run,
                    turns=_turn + 1,
                )

            # Execute each bash command
            messages.append({"role": "assistant", "content": resp.content})
            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                cmd = block.input.get("command", "echo 'no command'")
                commands_run.append({"command": cmd})
                try:
                    result = subprocess.run(  # noqa: S603
                        ["bash", "-c", cmd],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    output = result.stdout + result.stderr
                    commands_run[-1]["output"] = output[:500]
                except subprocess.TimeoutExpired:
                    output = "Command timed out after 10s"
                    commands_run[-1]["output"] = output
                except Exception as e:
                    output = f"Error: {e}"
                    commands_run[-1]["output"] = output

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:2000],
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        elapsed = int((time.monotonic() - start) * 1000)
        return _ok(
            f"Completed {max_turns} turns",
            elapsed,
            commands_run=commands_run,
            turns=max_turns,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return _err(str(e), elapsed)

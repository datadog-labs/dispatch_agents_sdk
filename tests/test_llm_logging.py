"""Tests for LLM logging functions in the SDK.

These tests verify the helper functions that extract fields from
OpenAI and Anthropic response objects, as well as the logging functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dispatch_agents.llm import (
    _extract_anthropic_response,
    _extract_openai_response,
    _is_anthropic_response,
    _is_openai_response,
    log_anthropic_response,
    log_llm_call,
    log_openai_response,
    log_response,
)

# =============================================================================
# Mock OpenAI Response Objects
# =============================================================================


def create_mock_openai_response(
    content: str | None = "Hello! How can I help?",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 8,
    finish_reason: str = "stop",
    tool_calls: list | None = None,
) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response.

    Note: We use spec=[] to ensure the mock doesn't have attributes
    that Anthropic responses would have (like `stop_reason`).
    """
    mock_response = MagicMock(spec=["choices", "usage", "model"])

    # Usage - needs spec to NOT have input_tokens (Anthropic attr)
    mock_response.usage = MagicMock(spec=["prompt_tokens", "completion_tokens"])
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens

    # Model
    mock_response.model = model

    # Message
    mock_message = MagicMock(spec=["content", "tool_calls"])
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    # Choice
    mock_choice = MagicMock(spec=["message", "finish_reason"])
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response.choices = [mock_choice]

    return mock_response


def create_mock_openai_tool_call(
    id: str = "call_123",
    name: str = "get_weather",
    arguments: str = '{"location": "NYC"}',
) -> MagicMock:
    """Create a mock OpenAI tool call."""
    mock_tc = MagicMock(spec=["id", "type", "function"])
    mock_tc.id = id
    mock_tc.type = "function"
    mock_tc.function = MagicMock(spec=["name", "arguments"])
    mock_tc.function.name = name
    mock_tc.function.arguments = arguments
    return mock_tc


# =============================================================================
# Mock Anthropic Response Objects
# =============================================================================


def create_mock_anthropic_response(
    text: str | None = "Hello! I'm Claude.",
    model: str = "claude-3-5-sonnet-20241022",
    input_tokens: int = 10,
    output_tokens: int = 8,
    stop_reason: str = "end_turn",
    tool_use_blocks: list | None = None,
) -> MagicMock:
    """Create a mock Anthropic Message response.

    Note: We use spec=[] to ensure the mock doesn't have attributes
    that OpenAI responses would have (like `choices`).
    """
    mock_response = MagicMock(spec=["content", "usage", "model", "stop_reason"])

    # Usage - needs spec to NOT have prompt_tokens (OpenAI attr)
    mock_response.usage = MagicMock(spec=["input_tokens", "output_tokens"])
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens

    # Model
    mock_response.model = model

    # Stop reason
    mock_response.stop_reason = stop_reason

    # Content blocks
    content_blocks = []
    if text is not None:
        text_block = MagicMock(spec=["text", "type"])
        text_block.text = text
        text_block.type = "text"
        content_blocks.append(text_block)

    if tool_use_blocks:
        content_blocks.extend(tool_use_blocks)

    mock_response.content = content_blocks

    return mock_response


def create_mock_anthropic_tool_use(
    id: str = "toolu_123",
    name: str = "get_weather",
    input_data: dict | str | None = None,
) -> MagicMock:
    """Create a mock Anthropic tool_use block.

    Note: spec ensures it doesn't have `text` attribute (which text blocks have).
    """
    if input_data is None:
        input_data = {"location": "NYC"}
    mock_tool = MagicMock(spec=["id", "type", "name", "input"])
    mock_tool.id = id
    mock_tool.type = "tool_use"
    mock_tool.name = name
    mock_tool.input = input_data
    return mock_tool


# =============================================================================
# Tests: OpenAI Response Detection
# =============================================================================


class TestIsOpenAIResponse:
    """Test _is_openai_response detection function."""

    def test_valid_openai_response(self):
        """Test detection of valid OpenAI response."""
        response = create_mock_openai_response()
        assert _is_openai_response(response) is True

    def test_missing_choices(self):
        """Test response without choices attribute."""
        response = MagicMock(spec=[])
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.model = "gpt-4o"
        assert _is_openai_response(response) is False

    def test_missing_usage(self):
        """Test response without usage attribute."""
        response = MagicMock(spec=[])
        response.choices = []
        response.model = "gpt-4o"
        assert _is_openai_response(response) is False

    def test_anthropic_response_not_detected(self):
        """Test that Anthropic response is not detected as OpenAI."""
        response = create_mock_anthropic_response()
        assert _is_openai_response(response) is False


# =============================================================================
# Tests: Anthropic Response Detection
# =============================================================================


class TestIsAnthropicResponse:
    """Test _is_anthropic_response detection function."""

    def test_valid_anthropic_response(self):
        """Test detection of valid Anthropic response."""
        response = create_mock_anthropic_response()
        assert _is_anthropic_response(response) is True

    def test_missing_content(self):
        """Test response without content attribute."""
        response = MagicMock(spec=[])
        response.usage = MagicMock()
        response.usage.input_tokens = 10
        response.stop_reason = "end_turn"
        assert _is_anthropic_response(response) is False

    def test_missing_stop_reason(self):
        """Test response without stop_reason attribute."""
        response = MagicMock(spec=[])
        response.content = []
        response.usage = MagicMock()
        response.usage.input_tokens = 10
        assert _is_anthropic_response(response) is False

    def test_openai_response_not_detected(self):
        """Test that OpenAI response is not detected as Anthropic."""
        response = create_mock_openai_response()
        assert _is_anthropic_response(response) is False


# =============================================================================
# Tests: OpenAI Response Extraction
# =============================================================================


class TestExtractOpenAIResponse:
    """Test _extract_openai_response function."""

    def test_basic_extraction(self):
        """Test extracting basic fields from OpenAI response."""
        response = create_mock_openai_response(
            content="Hello there!",
            model="gpt-4o-mini",
            prompt_tokens=15,
            completion_tokens=5,
        )

        extracted = _extract_openai_response(response)

        assert extracted["response_content"] == "Hello there!"
        assert extracted["model"] == "gpt-4o-mini"
        assert extracted["provider"] == "openai"
        assert extracted["input_tokens"] == 15
        assert extracted["output_tokens"] == 5
        assert extracted["finish_reason"] == "stop"
        assert extracted["tool_calls"] is None

    def test_extraction_with_tool_calls(self):
        """Test extracting response with tool calls."""
        tool_call = create_mock_openai_tool_call(
            id="call_abc",
            name="search",
            arguments='{"query": "weather"}',
        )
        response = create_mock_openai_response(
            content=None,  # Tool calls often have no content
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        extracted = _extract_openai_response(response)

        assert extracted["response_content"] is None
        assert extracted["finish_reason"] == "tool_calls"
        assert extracted["tool_calls"] is not None
        assert len(extracted["tool_calls"]) == 1
        assert extracted["tool_calls"][0]["id"] == "call_abc"
        assert extracted["tool_calls"][0]["function"]["name"] == "search"
        assert (
            extracted["tool_calls"][0]["function"]["arguments"]
            == '{"query": "weather"}'
        )

    def test_extraction_empty_choices(self):
        """Test extraction when choices is empty."""
        response = MagicMock()
        response.choices = []
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 0
        response.model = "gpt-4o"

        extracted = _extract_openai_response(response)

        assert extracted["response_content"] is None
        assert extracted["finish_reason"] == "stop"


# =============================================================================
# Tests: Anthropic Response Extraction
# =============================================================================


class TestExtractAnthropicResponse:
    """Test _extract_anthropic_response function."""

    def test_basic_extraction(self):
        """Test extracting basic fields from Anthropic response."""
        response = create_mock_anthropic_response(
            text="Hello! I'm Claude.",
            model="claude-3-5-sonnet-20241022",
            input_tokens=20,
            output_tokens=10,
        )

        extracted = _extract_anthropic_response(response)

        assert extracted["response_content"] == "Hello! I'm Claude."
        assert extracted["model"] == "claude-3-5-sonnet-20241022"
        assert extracted["provider"] == "anthropic"
        assert extracted["input_tokens"] == 20
        assert extracted["output_tokens"] == 10
        assert extracted["finish_reason"] == "stop"  # end_turn -> stop
        assert extracted["tool_calls"] is None

    def test_extraction_with_tool_use(self):
        """Test extracting response with tool_use blocks."""
        tool_use = create_mock_anthropic_tool_use(
            id="toolu_xyz",
            name="calculator",
            input_data={"expression": "2+2"},
        )
        response = create_mock_anthropic_response(
            text=None,  # May have no text with tool use
            stop_reason="tool_use",
            tool_use_blocks=[tool_use],
        )

        extracted = _extract_anthropic_response(response)

        assert extracted["finish_reason"] == "tool_calls"  # tool_use -> tool_calls
        assert extracted["tool_calls"] is not None
        assert len(extracted["tool_calls"]) == 1
        assert extracted["tool_calls"][0]["id"] == "toolu_xyz"
        assert extracted["tool_calls"][0]["function"]["name"] == "calculator"

    def test_stop_reason_mapping(self):
        """Test that Anthropic stop_reason is mapped correctly."""
        test_cases = [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
        ]

        for anthropic_reason, expected_reason in test_cases:
            response = create_mock_anthropic_response(stop_reason=anthropic_reason)
            extracted = _extract_anthropic_response(response)
            assert extracted["finish_reason"] == expected_reason, (
                f"Failed for {anthropic_reason}"
            )

    def test_multiple_text_blocks(self):
        """Test extraction with multiple text blocks."""
        response = MagicMock()
        response.model = "claude-3-5-sonnet-20241022"
        response.stop_reason = "end_turn"
        response.usage = MagicMock()
        response.usage.input_tokens = 10
        response.usage.output_tokens = 20

        # Multiple text blocks
        block1 = MagicMock()
        block1.text = "First part."
        block1.type = "text"

        block2 = MagicMock()
        block2.text = "Second part."
        block2.type = "text"

        response.content = [block1, block2]

        extracted = _extract_anthropic_response(response)
        assert extracted["response_content"] == "First part.\nSecond part."


# =============================================================================
# Tests: log_llm_call Function
# =============================================================================


class TestLogLLMCall:
    """Test the log_llm_call function."""

    @pytest.fixture
    def mock_httpx_post(self):
        """Mock httpx.AsyncClient.post for API calls."""
        with patch("dispatch_agents.llm.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "llm_call_id": "call-123",
                "cost_usd": 0.00025,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm._get_api_base_url")
    @patch("dispatch_agents.llm._get_auth_headers")
    @patch("dispatch_agents.llm.get_current_trace_id")
    @patch("dispatch_agents.llm.get_current_invocation_id")
    async def test_log_llm_call_basic(
        self,
        mock_inv_id,
        mock_trace_id,
        mock_headers,
        mock_base_url,
        mock_httpx_post,
    ):
        """Test basic LLM call logging."""
        mock_base_url.return_value = "http://localhost:8000/api/unstable/namespace/test"
        mock_headers.return_value = {"Authorization": "Bearer test-token"}
        mock_trace_id.return_value = "auto-trace-123"
        mock_inv_id.return_value = "auto-inv-456"

        result = await log_llm_call(
            input_messages=[{"role": "user", "content": "Hello"}],
            response_content="Hi there!",
            model="gpt-4o",
            provider="openai",
            input_tokens=5,
            output_tokens=3,
        )

        assert result == "call-123"

        # Verify the API call
        mock_httpx_post.post.assert_called_once()
        call_args = mock_httpx_post.post.call_args
        assert "llm/log" in call_args.args[0]

        payload = call_args.kwargs["json"]
        assert payload["model"] == "gpt-4o"
        assert payload["provider"] == "openai"
        assert payload["input_tokens"] == 5
        assert payload["output_tokens"] == 3
        assert payload["trace_id"] == "auto-trace-123"
        assert payload["invocation_id"] == "auto-inv-456"

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm._get_api_base_url")
    @patch("dispatch_agents.llm._get_auth_headers")
    @patch("dispatch_agents.llm.get_current_trace_id")
    @patch("dispatch_agents.llm.get_current_invocation_id")
    async def test_log_llm_call_with_explicit_ids(
        self,
        mock_inv_id,
        mock_trace_id,
        mock_headers,
        mock_base_url,
        mock_httpx_post,
    ):
        """Test logging with explicit trace_id and invocation_id."""
        mock_base_url.return_value = "http://localhost:8000/api/unstable/namespace/test"
        mock_headers.return_value = {"Authorization": "Bearer test-token"}
        mock_trace_id.return_value = None  # Would be auto-detected
        mock_inv_id.return_value = None

        await log_llm_call(
            input_messages=[{"role": "user", "content": "Hello"}],
            response_content="Hi!",
            model="gpt-4o",
            provider="openai",
            input_tokens=5,
            output_tokens=2,
            trace_id="explicit-trace",
            invocation_id="explicit-inv",
        )

        payload = mock_httpx_post.post.call_args.kwargs["json"]
        assert payload["trace_id"] == "explicit-trace"
        assert payload["invocation_id"] == "explicit-inv"

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm._get_api_base_url")
    @patch("dispatch_agents.llm._get_auth_headers")
    @patch("dispatch_agents.llm.get_current_trace_id")
    @patch("dispatch_agents.llm.get_current_invocation_id")
    async def test_log_llm_call_with_tool_calls(
        self,
        mock_inv_id,
        mock_trace_id,
        mock_headers,
        mock_base_url,
        mock_httpx_post,
    ):
        """Test logging with tool calls."""
        mock_base_url.return_value = "http://localhost:8000/api/unstable/namespace/test"
        mock_headers.return_value = {"Authorization": "Bearer test-token"}
        mock_trace_id.return_value = None
        mock_inv_id.return_value = None

        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]

        await log_llm_call(
            input_messages=[{"role": "user", "content": "Weather?"}],
            response_content=None,
            model="gpt-4o",
            provider="openai",
            input_tokens=5,
            output_tokens=10,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )

        payload = mock_httpx_post.post.call_args.kwargs["json"]
        assert payload["tool_calls"] == tool_calls
        assert payload["finish_reason"] == "tool_calls"


# =============================================================================
# Tests: Helper Logging Functions
# =============================================================================


class TestLogOpenAIResponse:
    """Test log_openai_response convenience function."""

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm.log_llm_call")
    async def test_log_openai_response(self, mock_log_call):
        """Test that log_openai_response extracts and logs correctly."""
        mock_log_call.return_value = "call-123"

        response = create_mock_openai_response(
            content="Test response",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
        )
        messages = [{"role": "user", "content": "Test"}]

        result = await log_openai_response(messages, response)

        assert result == "call-123"
        mock_log_call.assert_called_once()
        call_kwargs = mock_log_call.call_args.kwargs

        assert call_kwargs["input_messages"] == messages
        assert call_kwargs["response_content"] == "Test response"
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["input_tokens"] == 10
        assert call_kwargs["output_tokens"] == 5


class TestLogAnthropicResponse:
    """Test log_anthropic_response convenience function."""

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm.log_llm_call")
    async def test_log_anthropic_response(self, mock_log_call):
        """Test that log_anthropic_response extracts and logs correctly."""
        mock_log_call.return_value = "call-456"

        response = create_mock_anthropic_response(
            text="Test from Claude",
            model="claude-3-5-sonnet-20241022",
            input_tokens=15,
            output_tokens=8,
        )
        messages = [{"role": "user", "content": "Test"}]

        result = await log_anthropic_response(messages, response)

        assert result == "call-456"
        mock_log_call.assert_called_once()
        call_kwargs = mock_log_call.call_args.kwargs

        assert call_kwargs["input_messages"] == messages
        assert call_kwargs["response_content"] == "Test from Claude"
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["input_tokens"] == 15
        assert call_kwargs["output_tokens"] == 8


class TestLogResponse:
    """Test log_response auto-detection function."""

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm.log_openai_response")
    async def test_auto_detects_openai(self, mock_log_openai):
        """Test that OpenAI response is auto-detected."""
        mock_log_openai.return_value = "call-openai"

        response = create_mock_openai_response()
        messages = [{"role": "user", "content": "Test"}]

        result = await log_response(messages, response)

        assert result == "call-openai"
        mock_log_openai.assert_called_once()

    @pytest.mark.asyncio
    @patch("dispatch_agents.llm.log_anthropic_response")
    async def test_auto_detects_anthropic(self, mock_log_anthropic):
        """Test that Anthropic response is auto-detected."""
        mock_log_anthropic.return_value = "call-anthropic"

        response = create_mock_anthropic_response()
        messages = [{"role": "user", "content": "Test"}]

        result = await log_response(messages, response)

        assert result == "call-anthropic"
        mock_log_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_response_raises_error(self):
        """Test that unknown response type raises ValueError."""
        unknown_response = MagicMock(spec=[])  # Empty spec, no attributes
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(ValueError, match="Unrecognized response type"):
            await log_response(messages, unknown_response)

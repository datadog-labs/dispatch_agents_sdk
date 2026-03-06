"""LangChain adapter for Dispatch LLM Gateway.

Provides a LangChain-compatible ChatModel that uses the Dispatch LLM proxy
for automatic trace correlation and cost tracking.

Example:
    from dispatch_agents.llm_langchain import ChatDispatch

    # Use as a drop-in replacement for ChatOpenAI
    llm = ChatDispatch(model="gpt-4o")

    # Works with LangGraph
    agent = create_react_agent(llm, tools=tools)

    # Works with structured output
    structured_llm = llm.with_structured_output(MySchema)
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from .llm import LLMClient, LLMResponse


def _get_tool_schema(tool: Any) -> dict[str, Any]:
    """Extract JSON schema from a tool's args_schema, handling both Pydantic models and dicts."""
    if not hasattr(tool, "args_schema") or not tool.args_schema:
        return {}

    args_schema = tool.args_schema

    # If it's already a dict, return it directly
    if isinstance(args_schema, dict):
        return args_schema

    # If it's a Pydantic model class, call model_json_schema()
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()

    # Fallback: try to convert to dict
    return {}


def _convert_tools_to_openai_format(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert various tool formats to OpenAI function calling format.

    Handles:
    - LangChain BaseTool objects
    - MCP tools (with args_schema as dict)
    - Raw dicts already in OpenAI format
    - Callable functions with docstrings
    """
    formatted_tools = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            # LangChain BaseTool or MCP tool
            tool_schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                },
            }
            # Get parameters schema, handling both Pydantic models and dicts
            params = _get_tool_schema(tool)
            if params:
                tool_schema["function"]["parameters"] = params
            formatted_tools.append(tool_schema)
        elif isinstance(tool, dict):
            # Already in OpenAI format
            formatted_tools.append(tool)
        elif callable(tool):
            # Function with docstring
            import inspect

            sig = inspect.signature(tool)
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": tool.__doc__ or "",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                name: {"type": "string"} for name in sig.parameters
                            },
                        },
                    },
                }
            )
    return formatted_tools


def _serialize_tool_arguments(args: Any) -> str:
    """Serialize tool call arguments to a JSON string.

    OpenAI-compatible APIs expect tool call arguments as a JSON string,
    not a Python dict string representation.
    """
    import json

    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        return json.dumps(args)
    # Fallback: try to convert to dict then serialize
    try:
        return json.dumps(dict(args))
    except (TypeError, ValueError):
        return json.dumps({})


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dict for the LLM API."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg_dict: dict[str, Any] = {"role": "assistant", "content": message.content}
        # Include tool calls if present
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _serialize_tool_arguments(tc["args"]),
                    },
                }
                for tc in message.tool_calls
            ]
        return msg_dict
    elif isinstance(message, ToolMessage):
        # tool_call_id is required by OpenAI-compatible APIs - use message id as fallback
        tool_call_id = message.tool_call_id
        if not tool_call_id:
            # Fallback: use message id or generate one
            tool_call_id = getattr(message, "id", None) or f"call_{id(message)}"
        return {
            "role": "tool",
            "content": str(message.content) if message.content else "",
            "tool_call_id": tool_call_id,
        }
    else:
        # Fallback for other message types
        return {"role": "user", "content": str(message.content)}


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Parse tool call arguments to a dict.

    OpenAI returns arguments as a JSON string, but LangChain expects a dict.
    Handles various input formats gracefully.
    """
    import json

    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments or arguments.strip() == "":
            return {}
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _convert_response_to_message(response: LLMResponse) -> AIMessage:
    """Convert an LLM response to a LangChain AIMessage."""
    # Build tool_calls list if present
    tool_calls = []
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": _parse_tool_arguments(tc.function.arguments),
                }
            )

    # Build response metadata
    response_metadata = {
        "llm_call_id": response.llm_call_id,
        "model": response.model,
        "provider": response.provider,
        "finish_reason": response.finish_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
    }
    if response.variant_name:
        response_metadata["variant_name"] = response.variant_name

    # Build usage metadata
    usage_metadata = {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.input_tokens + response.output_tokens,
    }

    return AIMessage(
        content=response.content or "",
        tool_calls=tool_calls if tool_calls else [],
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
    )


class ChatDispatch(BaseChatModel):
    """LangChain ChatModel that uses Dispatch LLM Gateway.

    Provides automatic trace correlation and cost tracking for all LLM calls
    when used within a Dispatch agent handler.

    Example:
        from dispatch_agents.llm_langchain import ChatDispatch

        # Basic usage
        llm = ChatDispatch(model="gpt-4o")
        response = await llm.ainvoke([HumanMessage(content="Hello!")])

        # With LangGraph ReAct agent
        agent = create_react_agent(llm, tools=my_tools)

        # With structured output
        structured_llm = llm.with_structured_output(MyPydanticModel)
        result = await structured_llm.ainvoke([...])

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet").
               Uses org default if not specified.
        provider: Provider name (e.g., "openai", "anthropic").
                  Uses org default if not specified.
        temperature: Sampling temperature (0-2), default 1.0
        max_tokens: Maximum tokens in response (optional)
    """

    model: str | None = Field(default=None, description="Model name to use")
    provider: str | None = Field(default=None, description="Provider name")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Max tokens in response")

    _client: LLMClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = LLMClient()

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "dispatch"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for caching."""
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - wraps async implementation."""
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, need to run in a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._agenerate(messages, stop, run_manager, **kwargs),  # type: ignore[arg-type]
                )
                return future.result()
        else:
            # No running loop, we can use asyncio.run
            return asyncio.run(
                self._agenerate(messages, stop, run_manager, **kwargs)  # type: ignore[arg-type]
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation using Dispatch LLM Gateway."""
        if self._client is None:
            self._client = LLMClient()

        # Convert messages to API format
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Extract tools from kwargs if provided (for function calling)
        tools = kwargs.get("tools")
        if tools:
            # Convert tools to OpenAI format, handling various tool types
            formatted_tools = _convert_tools_to_openai_format(tools)
            tools = formatted_tools if formatted_tools else None

        # Call the LLM Gateway
        response = await self._client.inference(
            messages=message_dicts,
            model=self.model,
            provider=self.provider,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Convert response to LangChain format
        ai_message = _convert_response_to_message(response)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(
            generations=[generation],
            llm_output={
                "model": response.model,
                "provider": response.provider,
                "llm_call_id": response.llm_call_id,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
            },
        )

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> "ChatDispatch":
        """Bind tools to this model for function calling.

        Args:
            tools: List of tools (LangChain tools, MCP tools, functions, or tool dicts)

        Returns:
            A new ChatDispatch instance with tools bound
        """
        # Convert tools to OpenAI format, handling various tool types
        formatted_tools = _convert_tools_to_openai_format(tools)

        # Return a new instance that will pass tools in kwargs.
        # .bind() returns Runnable, not ChatDispatch, but callers expect ChatDispatch.
        bound = (
            self.__class__(
                model=self.model,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            .configurable_fields(
                # Store tools in config to be passed through
            )
            .bind(tools=formatted_tools, **kwargs)
        )
        return bound  # type: ignore[return-value]

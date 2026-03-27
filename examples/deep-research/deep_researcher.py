"""Main Deep Research orchestrator using OpenAI SDK tool-calling."""

import asyncio
import json
from typing import Any

from configuration import Configuration, SearchAPI
from openai import AsyncOpenAI
from prompts import (
    CLARIFY_WITH_USER_PROMPT,
    LEAD_RESEARCHER_PROMPT,
    RESEARCHER_PROMPT,
    TRANSFORM_MESSAGES_TO_RESEARCH_TOPIC_PROMPT,
)
from state import (
    AssistantMessage,
    ClarifyWithUser,
    ResearchContext,
    ResearcherContext,
    ResearchQuestion,
    ToolCall,
    ToolCallFunction,
    ToolResultMessage,
)
from tools import (
    compress_research,
    filter_recent_messages,
    format_messages_as_string,
    generate_final_report,
    get_today_str,
    get_token_param,
    is_token_limit_exceeded,
    tavily_search,
)


class DeepResearcher:
    """Orchestrates the full research pipeline: brief → research → report."""

    def __init__(self, config: Configuration | None = None):
        self.config = config or Configuration()
        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set as an environment variable")
        self.client = AsyncOpenAI(api_key=self.config.openai_api_key)
        self.context = ResearchContext()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_result(tool_call_id: str, name: str, content: str) -> ToolResultMessage:
        return ToolResultMessage(
            tool_call_id=tool_call_id,
            name=name,
            content=content,
        )

    @staticmethod
    def _assistant_msg(message: Any) -> AssistantMessage:
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in message.tool_calls
            ]
        return AssistantMessage(
            content=message.content or "",
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_message: str) -> str:
        """Run the full pipeline and return the final report."""
        self.context.messages.append({"role": "user", "content": user_message})

        if self.config.allow_clarification:
            clarification = await self._clarify()
            if clarification.need_clarification:
                self.context.messages.append(
                    {"role": "assistant", "content": clarification.question}
                )
                return clarification.question
            self.context.messages.append(
                {"role": "assistant", "content": clarification.verification}
            )

        await self._write_brief()
        await self._conduct_research()
        report = await self._generate_report()
        self.context.final_report = report
        return report

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _clarify(self) -> ClarifyWithUser:
        response = await self.client.beta.chat.completions.parse(
            model=self.config.research_model,
            messages=[
                {
                    "role": "system",
                    "content": CLARIFY_WITH_USER_PROMPT.format(date=get_today_str()),
                },
                {
                    "role": "user",
                    "content": format_messages_as_string(self.context.messages),
                },
            ],
            response_format=ClarifyWithUser,
            **get_token_param(
                self.config.research_model, self.config.research_model_max_tokens
            ),
        )
        return response.choices[0].message.parsed

    async def _write_brief(self) -> None:
        response = await self.client.beta.chat.completions.parse(
            model=self.config.research_model,
            messages=[
                {
                    "role": "system",
                    "content": TRANSFORM_MESSAGES_TO_RESEARCH_TOPIC_PROMPT.format(
                        date=get_today_str()
                    ),
                },
                {
                    "role": "user",
                    "content": format_messages_as_string(self.context.messages),
                },
            ],
            response_format=ResearchQuestion,
            **get_token_param(
                self.config.research_model, self.config.research_model_max_tokens
            ),
        )
        rq = response.choices[0].message.parsed
        self.context.research_brief = rq.research_brief

        self.context.supervisor_messages = [
            {
                "role": "system",
                "content": LEAD_RESEARCHER_PROMPT.format(
                    date=get_today_str(),
                    max_concurrent_research_units=self.config.max_concurrent_research_units,
                    max_researcher_iterations=self.config.max_researcher_iterations,
                ),
            },
            {"role": "user", "content": self.context.research_brief},
        ]

    async def _conduct_research(self) -> None:
        while self.context.research_iterations < self.config.max_researcher_iterations:
            self.context.research_iterations += 1

            resp = await self.client.chat.completions.create(
                model=self.config.research_model,
                messages=self.context.supervisor_messages,
                tools=_SUPERVISOR_TOOLS,
                **get_token_param(
                    self.config.research_model, self.config.research_model_max_tokens
                ),
            )
            msg = resp.choices[0].message
            self.context.supervisor_messages.append(
                self._assistant_msg(msg).model_dump(exclude_none=True)
            )

            if not msg.tool_calls or any(
                tc.function.name == "research_complete" for tc in msg.tool_calls
            ):
                break

            await self._exec_supervisor_tools(msg.tool_calls)

    async def _exec_supervisor_tools(self, tool_calls) -> None:
        research_tasks: list[tuple] = []
        tool_results: list[dict] = []

        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "think":
                tool_results.append(
                    self._tool_result(
                        tc.id,
                        "think",
                        f"Reflection recorded: {args.get('reflection', '')}",
                    ).model_dump()
                )
            elif name == "conduct_research":
                topic = args.get("research_topic", "").strip()
                if not topic:
                    tool_results.append(
                        self._tool_result(
                            tc.id, "conduct_research", "Error: empty research_topic"
                        ).model_dump()
                    )
                    continue
                research_tasks.append((tc.id, topic))

        if research_tasks:
            allowed = research_tasks[: self.config.max_concurrent_research_units]
            overflow = research_tasks[self.config.max_concurrent_research_units :]

            results = await asyncio.gather(
                *[self._run_researcher(t) for _, t in allowed]
            )
            for (tc_id, _), result in zip(allowed, results):
                tool_results.append(
                    self._tool_result(tc_id, "conduct_research", result).model_dump()
                )
                self.context.notes.append(result)

            for tc_id, _ in overflow:
                tool_results.append(
                    self._tool_result(
                        tc_id,
                        "conduct_research",
                        "Error: max concurrent units exceeded",
                    ).model_dump()
                )

        self.context.supervisor_messages.extend(tool_results)

    async def _run_researcher(self, topic: str) -> str:
        ctx = ResearcherContext(research_topic=topic)
        ctx.researcher_messages = [
            {
                "role": "system",
                "content": RESEARCHER_PROMPT.format(
                    date=get_today_str(), mcp_prompt=self.config.mcp_prompt or ""
                ),
            },
            {"role": "user", "content": topic},
        ]

        while ctx.tool_call_iterations < self.config.max_react_tool_calls:
            ctx.tool_call_iterations += 1

            try:
                resp = await self.client.chat.completions.create(
                    model=self.config.research_model,
                    messages=ctx.researcher_messages,
                    tools=_RESEARCHER_TOOLS,
                    **get_token_param(
                        self.config.research_model,
                        self.config.research_model_max_tokens,
                    ),
                )
            except Exception as e:
                if is_token_limit_exceeded(e):
                    ctx.researcher_messages = filter_recent_messages(
                        ctx.researcher_messages, 10
                    )
                    try:
                        resp = await self.client.chat.completions.create(
                            model=self.config.research_model,
                            messages=ctx.researcher_messages,
                            tools=_RESEARCHER_TOOLS,
                            **get_token_param(
                                self.config.research_model,
                                self.config.research_model_max_tokens,
                            ),
                        )
                    except Exception:
                        break
                else:
                    break

            msg = resp.choices[0].message
            ctx.researcher_messages.append(
                self._assistant_msg(msg).model_dump(exclude_none=True)
            )

            if not msg.tool_calls:
                break

            tool_results = await self._exec_researcher_tools(msg.tool_calls)
            ctx.researcher_messages.extend([tr.model_dump() for tr in tool_results])

            if any(tc.function.name == "research_complete" for tc in msg.tool_calls):
                break

        compressed = await compress_research(ctx.researcher_messages, self.config)
        ctx.compressed_research = compressed
        return compressed

    async def _exec_researcher_tools(self, tool_calls) -> list[ToolResultMessage]:
        results: list[ToolResultMessage] = []
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "think":
                results.append(
                    self._tool_result(
                        tc.id,
                        "think",
                        f"Reflection recorded: {args.get('reflection', '')}",
                    )
                )
            elif name == "search":
                queries = args.get("queries", [])
                if isinstance(queries, str):
                    queries = [queries]
                if self.config.search_api == SearchAPI.TAVILY:
                    text = await tavily_search(queries, self.config, max_results=5)
                else:
                    text = "Error: search API not configured"
                results.append(self._tool_result(tc.id, "search", text))
            elif name == "research_complete":
                results.append(
                    self._tool_result(
                        tc.id, "research_complete", "Research marked as complete"
                    )
                )
        return results

    async def _generate_report(self) -> str:
        return await generate_final_report(
            research_brief=self.context.research_brief or "",
            messages=self.context.messages,
            findings=self.context.notes,
            config=self.config,
        )


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

_SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Record strategic reflections and planning during research",
            "parameters": {
                "type": "object",
                "properties": {"reflection": {"type": "string"}},
                "required": ["reflection"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conduct_research",
            "description": "Delegate a research task to a specialised sub-agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "research_topic": {
                        "type": "string",
                        "description": "Detailed topic description (at least a paragraph)",
                    }
                },
                "required": ["research_topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_complete",
            "description": "Signal that research is complete",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

_RESEARCHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Reflect on search results and plan next steps",
            "parameters": {
                "type": "object",
                "properties": {"reflection": {"type": "string"}},
                "required": ["reflection"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_complete",
            "description": "Signal that enough information has been gathered",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

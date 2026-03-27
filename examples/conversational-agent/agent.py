"""Conversational Agent — multi-turn chat with persistent memory.

Demonstrates:
- Dispatch LLM Gateway (llm.inference) — call any LLM without managing API keys
- Long-term Memory — maintain conversation history across invocations
- @fn() callable function

This agent acts as a helpful assistant that remembers previous messages in the
same session. Send multiple messages with the same session_id to continue a
conversation.
"""

import json
import logging

from dispatch_agents import BasePayload, fn, llm, memory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant. You give clear, concise answers and
ask clarifying questions when the user's request is ambiguous.

Guidelines:
- Be direct and avoid filler.
- Use markdown formatting when it helps (bullet lists, code blocks, tables).
- If you don't know something, say so rather than guessing.
- Remember context from earlier in the conversation.
"""


class ChatRequest(BasePayload):
    """Input payload for a chat message."""

    session_id: str
    message: str


class ChatResponse(BasePayload):
    """Output payload with the assistant's reply."""

    session_id: str
    reply: str


@fn()
async def chat(payload: ChatRequest) -> ChatResponse:
    """Send a message and get a response, with full conversation history.

    Pass the same session_id across calls to continue a conversation.
    The agent remembers all previous messages in that session.

    Example — first turn::

        {"session_id": "demo-1", "message": "What is Dispatch?"}

    Example — follow-up (same session)::

        {"session_id": "demo-1", "message": "How do I deploy an agent?"}
    """
    mem_key = f"conversation:{payload.session_id}"

    # 1. Load existing conversation history from long-term memory
    # TODO: Update to use typed memory once the SDK supports generics
    # (e.g. memory.long_term.get[ConversationHistory](mem_key=...))
    history: list[dict] = []
    result = await memory.long_term.get(mem_key=mem_key)
    if result and result.value:
        history = json.loads(result.value)

    # 2. Append the new user message
    history.append({"role": "user", "content": payload.message})

    # 3. Call the LLM with the full conversation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    response = await llm.inference(messages)

    # 4. Save updated history to long-term memory
    history.append({"role": "assistant", "content": response.content})
    await memory.long_term.add(mem_key=mem_key, mem_val=json.dumps(history))

    return ChatResponse(
        session_id=payload.session_id,
        reply=response.content or "",
    )

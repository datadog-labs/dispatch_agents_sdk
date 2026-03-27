"""State definitions and data structures for the Deep Research agent."""

from pydantic import BaseModel, Field


class ClarifyWithUser(BaseModel):
    """Schema for user clarification requests."""

    need_clarification: bool
    question: str
    verification: str


class ResearchQuestion(BaseModel):
    """Schema for a structured research brief."""

    research_brief: str


class Summary(BaseModel):
    """Schema for a webpage summary with key excerpts."""

    summary: str
    key_excerpts: str


# ---------------------------------------------------------------------------
# Message types for OpenAI API tool-calling
# ---------------------------------------------------------------------------


class ToolCallFunction(BaseModel):
    """Function details within a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A single tool call from the assistant."""

    id: str
    type: str = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    """An assistant message, optionally containing tool calls."""

    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class ToolResultMessage(BaseModel):
    """A tool result message returned to the model."""

    tool_call_id: str
    role: str = "tool"
    name: str
    content: str


# ---------------------------------------------------------------------------
# Context state
# ---------------------------------------------------------------------------


class ResearchContext(BaseModel):
    """Top-level state for the research pipeline."""

    messages: list[dict] = Field(default_factory=list)
    research_brief: str | None = None
    supervisor_messages: list[dict] = Field(default_factory=list)
    research_iterations: int = 0
    notes: list[str] = Field(default_factory=list)
    final_report: str | None = None

    class Config:
        arbitrary_types_allowed = True


class ResearcherContext(BaseModel):
    """State for an individual researcher sub-agent."""

    researcher_messages: list[dict] = Field(default_factory=list)
    research_topic: str
    tool_call_iterations: int = 0
    compressed_research: str | None = None

    class Config:
        arbitrary_types_allowed = True

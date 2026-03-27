"""Configuration for the Deep Research agent."""

import os
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SearchAPI(str, Enum):
    """Available search API providers."""

    TAVILY = "tavily"
    NONE = "none"


class Configuration(BaseModel):
    """Main configuration for the Deep Research agent."""

    # General
    max_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=False)
    max_concurrent_research_units: int = Field(default=5, ge=1, le=20)

    # Research
    search_api: SearchAPI = Field(default=SearchAPI.TAVILY)
    max_researcher_iterations: int = Field(default=6, ge=1, le=10)
    max_react_tool_calls: int = Field(default=10, ge=1, le=30)

    # Models — defaults are affordable and widely available
    research_model: str = Field(default="gpt-4o-mini")
    research_model_max_tokens: int = Field(default=10000)
    compression_model: str = Field(default="gpt-4o-mini")
    compression_model_max_tokens: int = Field(default=8192)
    final_report_model: str = Field(default="gpt-4o")
    final_report_model_max_tokens: int = Field(default=10000)
    summarization_model: str = Field(default="gpt-4o-mini")
    summarization_model_max_tokens: int = Field(default=8192)
    max_content_length: int = Field(default=50000, ge=1000, le=200000)

    # Timeouts
    api_timeout: int = Field(default=120, ge=10, le=600)
    summarization_timeout: int = Field(default=60, ge=10, le=300)

    # API keys (from environment)
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    tavily_api_key: str | None = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # MCP
    mcp_prompt: str | None = Field(default=None)

    @field_validator(
        "research_model_max_tokens",
        "compression_model_max_tokens",
        "final_report_model_max_tokens",
        "summarization_model_max_tokens",
    )
    @classmethod
    def validate_token_limits(cls, v, info):
        if v < 100:
            raise ValueError(f"{info.field_name} must be >= 100")
        if v > 200000:
            raise ValueError(f"{info.field_name} must be <= 200,000")
        return v

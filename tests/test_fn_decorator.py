"""Tests for @fn decorator and invoke() function."""

import pytest
from pydantic import Field

from dispatch_agents import (
    HANDLER_METADATA,
    REGISTERED_HANDLERS,
    BasePayload,
    fn,
)


# Test payload models
class CalculateRequest(BasePayload):
    """Request payload for calculation."""

    x: int = Field(description="First number")
    y: int = Field(description="Second number")


class CalculateResponse(BasePayload):
    """Response payload for calculation."""

    result: int = Field(description="Calculation result")


class SimpleRequest(BasePayload):
    """Simple request payload."""

    data: str


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear handler registries before each test."""
    REGISTERED_HANDLERS.clear()
    HANDLER_METADATA.clear()
    yield
    REGISTERED_HANDLERS.clear()
    HANDLER_METADATA.clear()


# ============================================================================
# Test: Basic @fn decorator
# ============================================================================


def test_fn_decorator_basic():
    """Test basic @fn decorator registration."""

    @fn()
    async def add_numbers(payload: CalculateRequest) -> CalculateResponse:
        """Add two numbers together."""
        return CalculateResponse(result=payload.x + payload.y)

    # Check function was registered
    assert "add_numbers" in REGISTERED_HANDLERS
    assert REGISTERED_HANDLERS["add_numbers"] == add_numbers

    # Check metadata was stored (now a HandlerMetadata Pydantic model)
    assert "add_numbers" in HANDLER_METADATA
    metadata = HANDLER_METADATA["add_numbers"]
    assert metadata.handler_name == "add_numbers"
    assert metadata.topics == []  # No topics for @fn
    assert metadata.input_schema is not None
    assert metadata.output_schema is not None
    assert metadata.handler_doc == "Add two numbers together."


# ============================================================================
# Test: @fn with custom name
# ============================================================================


def test_fn_decorator_custom_name():
    """Test @fn decorator with custom function name."""

    @fn(name="custom_add")
    async def add_numbers(payload: CalculateRequest) -> CalculateResponse:
        return CalculateResponse(result=payload.x + payload.y)

    # Check function was registered with custom name
    assert "custom_add" in REGISTERED_HANDLERS
    assert "add_numbers" not in REGISTERED_HANDLERS

    metadata = HANDLER_METADATA["custom_add"]
    assert metadata.handler_name == "custom_add"


# ============================================================================
# Test: @fn with no return type
# ============================================================================


def test_fn_decorator_no_return():
    """Test @fn decorator with no return annotation."""

    @fn()
    async def log_data(payload: SimpleRequest) -> None:
        """Log the data."""
        print(f"Logging: {payload.data}")

    assert "log_data" in REGISTERED_HANDLERS
    metadata = HANDLER_METADATA["log_data"]
    # output_model not stored in HandlerMetadata, only output_schema
    assert metadata.output_schema is None


# ============================================================================
# Test: Schema extraction
# ============================================================================


def test_fn_schema_extraction():
    """Test that schemas are properly extracted from @fn functions."""

    @fn()
    async def calculate(payload: CalculateRequest) -> CalculateResponse:
        """Perform a calculation."""
        return CalculateResponse(result=payload.x + payload.y)

    metadata = HANDLER_METADATA["calculate"]

    # Check input schema (access via attribute, not dict indexing)
    input_schema = metadata.input_schema
    assert input_schema is not None
    assert "properties" in input_schema
    assert "x" in input_schema["properties"]
    assert "y" in input_schema["properties"]

    # Check output schema
    output_schema = metadata.output_schema
    assert output_schema is not None
    assert "properties" in output_schema
    assert "result" in output_schema["properties"]


# ============================================================================
# Test: Missing input type annotation
# ============================================================================


def test_fn_missing_input_annotation():
    """Test that @fn without input annotation raises an error."""

    with pytest.raises(ValueError, match="must have a first parameter"):

        @fn()
        async def bad_function(payload) -> SimpleRequest:  # Missing type annotation
            return SimpleRequest(data="")


# ============================================================================
# Test: Duplicate handler registration
# ============================================================================


def test_fn_duplicate_registration():
    """Test that registering the same handler name twice raises an error."""

    @fn()
    async def my_function(payload: SimpleRequest) -> None:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @fn()
        async def my_function(payload: SimpleRequest) -> None:  # noqa: F811
            pass


# ============================================================================
# Test: Metadata on function
# ============================================================================


def test_fn_metadata_on_function():
    """Test that metadata is attached to the function object."""

    @fn()
    async def my_func(payload: SimpleRequest) -> SimpleRequest:
        """My function doc."""
        return SimpleRequest(data="response")

    # Check _dispatch_metadata attribute (now a HandlerMetadata Pydantic model)
    assert hasattr(my_func, "_dispatch_metadata")
    metadata = my_func._dispatch_metadata  # type: ignore
    assert metadata.handler_name == "my_func"
    assert metadata.topics == []
    assert metadata.handler_doc == "My function doc."

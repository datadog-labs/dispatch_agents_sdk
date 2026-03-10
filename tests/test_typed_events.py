"""Tests for typed event handlers with Pydantic payloads."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from dispatch_agents import (
    HANDLER_METADATA,
    REGISTERED_HANDLERS,
    TOPIC_HANDLERS,
    BasePayload,
    dispatch_message,
    fn,
    get_handler_metadata,
    get_handler_schemas,
    on,
)
from dispatch_agents.models import (
    ErrorPayload,
    FunctionMessage,
    SuccessPayload,
    TopicMessage,
)


# Test payload models
class TestInputPayload(BasePayload):
    """Test input payload."""

    value: int = Field(description="A test integer value")
    name: str = Field(description="A test string value")


class TestOutputPayload(BasePayload):
    """Test output payload."""

    result: int = Field(description="The computed result")
    message: str = Field(description="A result message")


class SimplePayload(BasePayload):
    """Simple payload for basic tests."""

    data: str


@pytest.fixture(autouse=True)
def clear_triggers():
    """Clear registries before each test."""
    REGISTERED_HANDLERS.clear()
    HANDLER_METADATA.clear()
    TOPIC_HANDLERS.clear()
    yield
    REGISTERED_HANDLERS.clear()
    HANDLER_METADATA.clear()
    TOPIC_HANDLERS.clear()


# ============================================================================
# Test: Basic typed handler
# ============================================================================


@pytest.mark.asyncio
async def test_typed_handler_basic():
    """Test basic typed handler with input and output payloads."""

    @on(topic="test.basic")
    async def handler(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(
            result=payload.value * 2, message=f"Processed {payload.name}"
        )

    # Create test message
    message = TopicMessage.create(
        topic="test.basic",
        payload={"value": 5, "name": "test"},
        sender_id="test-sender",
    )

    # Dispatch
    result = await dispatch_message(message)

    # Verify - dispatch_message now returns SuccessPayload or ErrorPayload
    assert isinstance(result, SuccessPayload)
    assert result.result == {"result": 10, "message": "Processed test"}


# ============================================================================
# Test: Handler without return value
# ============================================================================


@pytest.mark.asyncio
async def test_typed_handler_no_return():
    """Test typed handler that doesn't return a value."""

    @on(topic="test.no_return")
    async def handler(payload: SimplePayload) -> None:
        print(f"Received: {payload.data}")
        return None

    # Create test message
    message = TopicMessage.create(
        topic="test.no_return", payload={"data": "hello"}, sender_id="test-sender"
    )

    # Dispatch
    result = await dispatch_message(message)

    # Verify - None return becomes SuccessPayload with result=None
    assert isinstance(result, SuccessPayload)
    assert result.result is None


# ============================================================================
# Test: Validation errors
# ============================================================================


@pytest.mark.asyncio
async def test_validation_error():
    """Test that validation errors are properly handled."""

    @on(topic="test.validation")
    async def handler(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(result=payload.value * 2, message="ok")

    # Create message with invalid payload (missing required field)
    message = TopicMessage.create(
        topic="test.validation",
        payload={"value": "not_an_int"},  # Invalid: should be int
        sender_id="test-sender",
    )

    # Dispatch
    result = await dispatch_message(message)

    # Verify error response - now returns ErrorPayload with detailed message
    assert isinstance(result, ErrorPayload)
    assert "validation error" in result.error.lower()
    assert "value" in result.error  # The field that failed validation
    assert result.error_type == "ValidationError"
    assert result.details is not None
    assert isinstance(result.details, list)


# ============================================================================
# Test: Schema extraction
# ============================================================================


def test_schema_extraction():
    """Test that schemas are properly extracted from handlers."""

    @on(topic="test.schema")
    async def handler(payload: TestInputPayload) -> TestOutputPayload:
        """A test handler for schema extraction."""
        return TestOutputPayload(result=0, message="")

    # Check metadata was stored
    metadata = get_handler_metadata("test.schema")
    assert metadata is not None
    assert metadata.topics == ["test.schema"]
    assert metadata.handler_name == "handler"
    assert metadata.handler_doc == "A test handler for schema extraction."

    # Check input schema
    input_schema = metadata.input_schema
    assert input_schema is not None
    assert "properties" in input_schema
    assert "value" in input_schema["properties"]
    assert "name" in input_schema["properties"]

    # Check output schema
    output_schema = metadata.output_schema
    assert output_schema is not None
    assert "properties" in output_schema
    assert "result" in output_schema["properties"]
    assert "message" in output_schema["properties"]


def test_get_handler_schemas():
    """Test that all handler schemas can be retrieved."""

    @on(topic="test.schema1")
    async def handler1(payload: SimplePayload) -> SimplePayload:
        return SimplePayload(data="")

    @on(topic="test.schema2")
    async def handler2(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(result=0, message="")

    @fn()
    async def my_function(payload: SimplePayload) -> SimplePayload:
        return SimplePayload(data="")

    # Get all schemas (includes both @on and @fn handlers)
    schemas = get_handler_schemas()
    assert len(schemas) == 3
    assert "handler1" in schemas
    assert "handler2" in schemas
    assert "my_function" in schemas

    # Verify schema structure
    assert schemas["handler1"].topics == ["test.schema1"]
    assert schemas["handler2"].topics == ["test.schema2"]
    assert schemas["my_function"].topics == []  # @fn has no topic triggers


# ============================================================================
# Test: Missing input type annotation
# ============================================================================


def test_missing_input_annotation():
    """Test that handlers without input annotations raise an error."""

    with pytest.raises(ValueError, match="must have a first parameter"):

        @on(topic="test.missing")
        async def handler(payload) -> SimplePayload:  # Missing type annotation
            return SimplePayload(data="")


# ============================================================================
# Test: Handler exception handling
# ============================================================================


@pytest.mark.asyncio
async def test_handler_exception():
    """Test that exceptions in handlers are properly caught and returned."""

    @on(topic="test.exception")
    async def handler(payload: SimplePayload) -> SimplePayload:
        raise ValueError("Something went wrong!")

    # Create test message
    message = TopicMessage.create(
        topic="test.exception", payload={"data": "test"}, sender_id="test-sender"
    )

    # Dispatch
    result = await dispatch_message(message)

    # Verify error response - now returns ErrorPayload
    assert isinstance(result, ErrorPayload)
    assert "Something went wrong!" in result.error
    assert result.error_type == "ValueError"
    assert result.trace is not None


# ============================================================================
# Test: Duplicate topic registration
# ============================================================================


def test_multiple_handlers_same_topic():
    """Test that multiple handlers can register for the same topic (fan-out pattern)."""

    @on(topic="test.fanout")
    async def handler1(payload: SimplePayload) -> None:
        pass

    @on(topic="test.fanout")
    async def handler2(payload: SimplePayload) -> None:
        pass

    # Both handlers should be registered for the same topic
    assert "test.fanout" in TOPIC_HANDLERS
    assert "handler1" in TOPIC_HANDLERS["test.fanout"]
    assert "handler2" in TOPIC_HANDLERS["test.fanout"]
    assert len(TOPIC_HANDLERS["test.fanout"]) == 2


# ============================================================================
# Test: Unified registry is populated
# ============================================================================


def test_unified_registry_populated():
    """Test that unified registries are populated when handlers are registered."""

    @on(topic="test.registry")
    async def handler(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(result=0, message="")

    # Check TOPIC_HANDLERS (topic -> handler_name mapping)
    assert "test.registry" in TOPIC_HANDLERS
    assert "handler" in TOPIC_HANDLERS["test.registry"]

    # Check REGISTERED_HANDLERS (handler_name -> function)
    assert "handler" in REGISTERED_HANDLERS

    # Check HANDLER_METADATA (handler_name -> HandlerMetadata)
    assert "handler" in HANDLER_METADATA
    metadata = HANDLER_METADATA["handler"]
    assert metadata.topics == ["test.registry"]
    # input_schema/output_schema are stored, input_model/output_model extracted via get_type_hints
    assert metadata.input_schema is not None
    assert metadata.output_schema is not None


# ============================================================================
# Test: BasePayload alias
# ============================================================================


def test_basepayload_alias():
    """Test that BasePayload works correctly for event payloads."""

    # BasePayload inherits from StrictBaseModel (which inherits from BaseModel)
    # so it should work the same way
    class CustomPayload(BasePayload):
        field: str

    obj = CustomPayload(field="test")
    assert obj.field == "test"
    assert isinstance(obj, BaseModel)

    # Test that it rejects extra fields (strict validation)
    with pytest.raises(ValidationError):
        CustomPayload(field="test", extra="not_allowed")  # type: ignore[call-arg]


# ============================================================================
# Test: @fn decorator and FunctionMessage dispatch
# ============================================================================


def test_fn_decorator_registration():
    """Test that @fn decorator registers handlers correctly."""

    @fn()
    async def get_weather(payload: TestInputPayload) -> TestOutputPayload:
        """Get weather for a location."""
        return TestOutputPayload(result=payload.value, message=payload.name)

    # Check handler is registered by function name
    assert "get_weather" in REGISTERED_HANDLERS
    assert "get_weather" in HANDLER_METADATA

    # Check metadata (now a HandlerMetadata Pydantic model, not dict)
    metadata = HANDLER_METADATA["get_weather"]
    assert metadata.handler_name == "get_weather"
    assert metadata.topics == []  # @fn has no topic triggers
    assert metadata.input_schema is not None
    assert metadata.output_schema is not None
    assert metadata.handler_doc == "Get weather for a location."

    # NOT in TOPIC_HANDLERS (since it's a direct function, not topic-based)
    assert "get_weather" not in TOPIC_HANDLERS


def test_fn_decorator_with_custom_name():
    """Test @fn decorator with custom function name."""

    @fn(name="custom_function_name")
    async def internal_handler(payload: SimplePayload) -> SimplePayload:
        return SimplePayload(data=payload.data.upper())

    # Registered under custom name, not function name
    assert "custom_function_name" in REGISTERED_HANDLERS
    assert "custom_function_name" in HANDLER_METADATA
    assert "internal_handler" not in REGISTERED_HANDLERS


@pytest.mark.asyncio
async def test_function_message_dispatch():
    """Test dispatching a FunctionMessage to an @fn handler."""

    @fn()
    async def calculate(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(
            result=payload.value * 3, message=f"Calculated for {payload.name}"
        )

    # Create FunctionMessage
    message = FunctionMessage.create(
        function_name="calculate",
        payload={"value": 7, "name": "test"},
        sender_id="test-sender",
    )

    # Dispatch
    result = await dispatch_message(message)

    # Verify response - now returns SuccessPayload (same for all message types)
    assert isinstance(result, SuccessPayload)
    assert result.result == {"result": 21, "message": "Calculated for test"}


@pytest.mark.asyncio
async def test_function_message_dispatch_error():
    """Test that errors in @fn handlers return proper error responses."""

    @fn()
    async def failing_function(payload: SimplePayload) -> SimplePayload:
        raise RuntimeError("Something broke!")

    message = FunctionMessage.create(
        function_name="failing_function",
        payload={"data": "test"},
        sender_id="test-sender",
    )

    result = await dispatch_message(message)

    # Verify error response - now returns ErrorPayload
    assert isinstance(result, ErrorPayload)
    assert "Something broke!" in result.error
    assert result.error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_function_message_validation_error():
    """Test validation errors for FunctionMessage."""

    @fn()
    async def typed_function(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(result=0, message="")

    # Invalid payload - value should be int, not string
    message = FunctionMessage.create(
        function_name="typed_function",
        payload={"value": "not_an_int", "name": "test"},
        sender_id="test-sender",
    )

    result = await dispatch_message(message)

    # Verify error response - now returns ErrorPayload with detailed message
    assert isinstance(result, ErrorPayload)
    assert "validation error" in result.error.lower()
    assert "value" in result.error  # The field that failed validation
    assert result.error_type == "ValidationError"


def test_function_message_unknown_handler():
    """Test dispatching to unknown handler raises error."""

    message = FunctionMessage.create(
        function_name="nonexistent_function",
        payload={"data": "test"},
        sender_id="test-sender",
    )

    with pytest.raises(ValueError, match="No handler registered: nonexistent_function"):
        import asyncio

        asyncio.get_event_loop().run_until_complete(dispatch_message(message))


@pytest.mark.asyncio
async def test_on_handler_also_callable_by_name():
    """Test that @on handlers can also be invoked directly by function name.

    This verifies the unified registry design: @on handlers are registered
    in REGISTERED_HANDLERS and can be called via FunctionMessage.
    """

    @on(topic="user.created")
    async def handle_user_created(payload: TestInputPayload) -> TestOutputPayload:
        return TestOutputPayload(result=payload.value * 2, message="User created")

    # Can be invoked via topic
    topic_message = TopicMessage.create(
        topic="user.created",
        payload={"value": 5, "name": "alice"},
        sender_id="test-sender",
    )
    topic_result = await dispatch_message(topic_message)
    # dispatch_message now returns SuccessPayload regardless of message type
    assert isinstance(topic_result, SuccessPayload)
    assert topic_result.result == {"result": 10, "message": "User created"}

    # Can ALSO be invoked directly by function name
    fn_message = FunctionMessage.create(
        function_name="handle_user_created",
        payload={"value": 5, "name": "alice"},
        sender_id="test-sender",
    )
    fn_result = await dispatch_message(fn_message)
    # Same response type for both invocation methods
    assert isinstance(fn_result, SuccessPayload)
    assert fn_result.result == {"result": 10, "message": "User created"}

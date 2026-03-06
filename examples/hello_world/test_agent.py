"""Unit tests for hello_world agent handlers."""

import agent  # noqa: F401 - Import to register handlers
import pytest
from dispatch_agents import dispatch_message
from dispatch_agents.models import ErrorPayload, SuccessPayload, TopicMessage


@pytest.mark.asyncio
async def test_sleep_basic():
    """Test basic sleep handler with valid duration."""
    message = TopicMessage.create(
        topic="sleep",
        payload={"duration_seconds": 2},
        sender_id="test-sender",
    )

    result = await dispatch_message(message)

    assert isinstance(result, SuccessPayload)
    assert result.result == {"seconds_slept": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"duration_seconds": 0},
        {"duration_seconds": -5},
        {},
        {"duration_seconds": "not_an_int"},
    ],
)
async def test_sleep_validation(payload):
    """Test that invalid inputs are rejected."""
    message = TopicMessage.create(
        topic="sleep",
        payload=payload,
        sender_id="test-sender",
    )

    result = await dispatch_message(message)

    assert isinstance(result, ErrorPayload)
    assert result.error == "Validation error"

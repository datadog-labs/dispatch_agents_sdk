"""Unit tests for hello_world agent handlers."""

import pytest
from agent import SleepRequest, sleep
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_sleep_basic():
    """Test basic sleep handler with valid duration."""
    result = await sleep(SleepRequest(duration_seconds=1))

    assert result.seconds_slept == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"duration_seconds": 0},
        {"duration_seconds": -5},
        {},
        {"duration_seconds": "not_an_int"},
    ],
)
def test_sleep_validation(payload):
    """Test that invalid inputs are rejected."""
    with pytest.raises(ValidationError):
        SleepRequest.model_validate(payload)

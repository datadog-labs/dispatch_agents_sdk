"""Unit tests for hello_world agent handlers.

The ``@fn()``/``@on()`` decorators return the wrapped function unchanged, so
handlers are imported and called directly here using only the public SDK
surface. Input validation that the dispatch layer would perform happens at
payload construction (Pydantic), so invalid inputs are exercised there.
"""

import pytest
from agent import SleepRequest, SleepResponse, sleep
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_sleep_basic():
    """Valid input runs the handler and returns a typed response."""
    result = await sleep(SleepRequest(duration_seconds=1))

    assert isinstance(result, SleepResponse)
    assert result.seconds_slept == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"duration_seconds": 0},  # PositiveInt rejects 0
        {"duration_seconds": -5},
        {},  # missing required field
        {"duration_seconds": "not_an_int"},
    ],
)
def test_sleep_validation(payload):
    """Invalid input is rejected at payload construction."""
    with pytest.raises(ValidationError):
        SleepRequest(**payload)

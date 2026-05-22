# hello_world

The simplest possible Dispatch agent — a walkthrough of the core SDK patterns.

## What This Demonstrates

| Handler | Decorator | Pattern |
|---------|-----------|---------|
| `greet` | `@on(topic="test")` | Typed payloads, `ValueError` (no retry) vs `OSError` (auto-retry) |
| `sleep` | `@on(topic="sleep")` | Long-running async work with progress logging |
| `on_pr_review_comment` | `@on(github_event=...)` | GitHub webhook subscription |
| `reverse` | `@fn()` | Function callable by other agents via `invoke()` |
| `storage_write` / `storage_read` | `@fn()` | Persistent storage at `/data` |
| `test_egress` | `@fn()` | Outbound HTTP with network egress controls |

## Key Patterns

### Typed Payloads

Every handler declares its input and output as a `BasePayload` subclass.
Dispatch validates incoming events against the input schema automatically:

```python
from dispatch_agents import BasePayload, on
from pydantic import Field

class GreetingPayload(BasePayload):
    subject: str = Field(default="World", description="The name or subject to greet")

class GreetingResponse(BasePayload):
    greeting: str = Field(description="The greeting message")

@on(topic="test")
async def greet(payload: GreetingPayload) -> GreetingResponse:
    return GreetingResponse(greeting=f"Hello {payload.subject}")
```

### Error Handling

```python
# ValueError → not retried (use for validation / business logic errors)
if not payload.subject:
    raise ValueError("Missing required field 'subject'")

# OSError → automatically retried with exponential backoff
if payload.subject == "oops":
    raise OSError("Transient error — will be retried")
```

### Persistent Storage

```python
from dispatch_agents import get_data_dir

data_dir = get_data_dir()         # /data in production, temp dir in local dev
path = data_dir / "my_file.txt"
path.write_text("hello")
```

Writes outside the allowed path raise `DisallowedWriteError` in dev mode.

## How to Run

```bash
# Deploy to local environment (requires Tilt running)
cd examples/hello_world
dispatch agent deploy

# Test the greet handler
dispatch event publish --topic test --payload '{"subject": "World"}'

# Test the reverse function (callable via @fn)
dispatch function invoke --agent hello-world --function reverse --payload '{"text": "hello"}'

# Test persistent storage
dispatch function invoke --agent hello-world --function storage_write \
  --payload '{"key": "notes.txt", "value": "hello world"}'
dispatch function invoke --agent hello-world --function storage_read \
  --payload '{"key": "notes.txt"}'
```

## Files

- `agent.py` — all handlers with inline documentation
- `dispatch.yaml` — agent name, namespace, and configuration

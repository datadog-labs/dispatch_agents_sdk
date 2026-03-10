import json

import httpx
import pytest

from dispatch_agents import BasePayload, dispatch_message, memory, on
from dispatch_agents.models import SuccessPayload, TopicMessage


class GithubEventPayload(BasePayload):
    """Test payload for github events."""

    repo: str
    branch: str


@on(topic="github")
async def github_event(payload: GithubEventPayload) -> None:
    print(f"Github event: repo={payload.repo}, branch={payload.branch}")
    return None


@pytest.mark.parametrize("topic", ["github", "direct"])
async def test_event(topic: str):
    if topic == "github":
        resp = await dispatch_message(
            TopicMessage.create(
                topic="github",
                payload={"repo": "test", "branch": "main"},
                sender_id="test",
            )
        )
        assert resp is not None
        # dispatch_message now returns SuccessPayload or ErrorPayload
        assert isinstance(resp, SuccessPayload)
        # Handler returns None, so result is None
        assert resp.result is None


async def trigger_callback(data: dict) -> None:
    return None


@pytest.mark.parametrize(
    "data", [{"repo": "test", "branch": "main"}, {"repo": "test2", "branch": "main2"}]
)
async def test_callback(data: dict):
    await trigger_callback(data)


# ==============================
# Memory client demonstrations
# ==============================


@pytest.mark.asyncio
async def test_memory_long_term(monkeypatch):
    def responder(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if "/memory/long-term" in path and method == "PUT":
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload["agent_name"] == "agent-a"
            assert payload["key"] == "greeting"
            assert payload["value"] == "hello-world"
            return httpx.Response(200, json={"message": "ok"})

        if "/memory/long-term" in path and method == "GET":
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload["agent_name"] == "agent-a"
            assert payload["key"] == "greeting"
            return httpx.Response(
                200,
                json={"value": "hello-world"},
            )

        if "/memory/long-term" in path and method == "DELETE":
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload["agent_name"] == "agent-a"
            assert payload["key"] == "greeting"
            return httpx.Response(200, json={"message": "deleted"})

        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(responder)
    original_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setenv("BACKEND_URL", "http://mock")
    monkeypatch.setenv("DISPATCH_NAMESPACE", "test-namespace")
    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    add_resp = await memory.long_term.add(
        mem_key="greeting", mem_val="hello-world", agent_name="agent-a"
    )
    assert add_resp.message == "ok"

    get_resp = await memory.long_term.get(mem_key="greeting", agent_name="agent-a")
    assert get_resp.value == "hello-world"

    del_resp = await memory.long_term.delete(mem_key="greeting", agent_name="agent-a")
    assert del_resp.message == "deleted"


@pytest.mark.asyncio
async def test_memory_short_term(monkeypatch):
    def responder(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if "/memory/short-term" in path and method == "PUT":
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload["agent_name"] == "agent-b"
            assert payload["session_id"] == "s-1"
            assert payload["session_data"] == {"turn": 1}
            return httpx.Response(200, json={"message": "ok"})

        if "/memory/short-term" in path and method == "GET":
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload["agent_name"] == "agent-b"
            assert payload["session_id"] == "s-1"
            return httpx.Response(200, json={"session_data": {"turn": 1}})

        if "/memory/short-term" in path and method == "DELETE":
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload["agent_name"] == "agent-b"
            assert payload["session_id"] == "s-1"
            return httpx.Response(200, json={"message": "deleted"})

        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(responder)
    original_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setenv("BACKEND_URL", "http://mock")
    monkeypatch.setenv("DISPATCH_NAMESPACE", "test-namespace")
    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    add_resp = await memory.short_term.add(
        session_id="s-1", session_data={"turn": 1}, agent_name="agent-b"
    )
    assert add_resp.message == "ok"

    get_resp = await memory.short_term.get(session_id="s-1", agent_name="agent-b")
    assert get_resp.session_data == {"turn": 1}

    del_resp = await memory.short_term.delete(session_id="s-1", agent_name="agent-b")
    assert del_resp.message == "deleted"

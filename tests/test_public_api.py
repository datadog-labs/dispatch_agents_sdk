"""Public SDK import contract tests."""


def test_root_public_api_contract():
    import dispatch_agents

    expected = {
        "BasePayload",
        "DisallowedWriteError",
        "KVGetResponse",
        "KVListResponse",
        "KVMemoryRecord",
        "LLMFunctionCall",
        "LLMToolCall",
        "McpHttpServerConfig",
        "MemoryWriteResponse",
        "SessionGetResponse",
        "config",
        "emit_event",
        "extra_headers",
        "fn",
        "get_current_invocation_id",
        "get_current_parent_id",
        "get_current_trace_id",
        "get_data_dir",
        "get_mcp_client",
        "get_mcp_servers_config",
        "init",
        "invoke",
        "llm",
        "memory",
        "on",
    }

    assert set(dispatch_agents.__all__) == expected
    for name in expected:
        assert hasattr(dispatch_agents, name)


def test_root_does_not_expose_known_internals():
    import dispatch_agents

    internal_names = {
        "AgentServiceClient",
        "FeedbackSentiment",
        "FeedbackType",
        "FunctionMessage",
        "HandlerMetadata",
        "InvocationStatus",
        "LongTermMemoryClient",
        "MemoryClient",
        "Message",
        "ShortTermMemoryClient",
        "TopicMessage",
        "dispatch_message",
        "get_handler_metadata",
        "get_handler_schemas",
        "run_init_hook",
    }

    for name in internal_names:
        assert not hasattr(dispatch_agents, name)
        assert name not in dispatch_agents.__all__

# Datadog MCP agent
(requires dispatch-cli>=0.1.2)

Simple langgraph agent that uses Datadog mcp to answer queries

## Setup
1. Run `dispatch init`. It should run without intervention, since we already defined the entrypoint in pyproject.toml (agent_main:trigger)
2. Then `dispatch run` with the --args tag to pass environment variables needed for MCP and ai-proxy access
```sh
dd-auth -- bash -c 'dispatch run --args "-e DD_API_KEY=$DD_API_KEY -e DD_APP_KEY=$DD_APP_KEY -e AI_GATEWAY_TOKEN=$(ddtool auth token rapid-ai-platform --datacenter us1.staging.dog)"'
```
3. Finally use `dispatch chat` or `dispatch send-event` to use the agent:
```sh
dispatch chat
```

```sh
dispatch send-event --input "What dd dashboards do we have related to nlq?" --context '{"thread_id":"thread"}
```
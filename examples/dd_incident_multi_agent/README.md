# LangGraph Incident Investigation Agent

This is an advanced version of the `dd_incident_multi_agent` example that uses **LangGraph** for workflow orchestration and **dispatch agents** for Datadog MCP integration.

## Key Improvements

1. **LangGraph Workflow**: Replaces sequential agent calls with a proper state-based workflow graph
2. **Dispatch Agent Integration**: Uses `dispatch_agents.invoke()` to call the `dd_mcp_agent` instead of direct MCP client initialization
3. **Better State Management**: Centralized state tracking through LangGraph's state system
4. **Parallel Processing**: Hypothesis investigations run in parallel for better performance
5. **Thread Continuity**: Maintains conversation context with the MCP agent across multiple calls

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   LangGraph     │    │  Dispatch Agent  │    │    dd_mcp_agent     │
│   Workflow      │───▶│     Wrapper      │───▶│   (Datadog MCP)     │
│                 │    │   (Tool Layer)   │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Workflow Steps

1. **Gather Context**: Calls `dd_mcp_agent` to collect incident information, logs, metrics, and traces
2. **Generate Hypotheses**: Uses LLM to create 3 testable hypotheses based on the context
3. **Investigate Hypotheses**: Spawns parallel investigations, each calling `dd_mcp_agent` with hypothesis-specific queries
4. **Generate Report**: Compiles findings into a comprehensive markdown report

## Files

- `agent_main.py` - Main dispatch agent entry point
- `investigation_graph.py` - LangGraph workflow implementation
- `dispatch_agent_wrapper.py` - LangChain tool wrapper for dispatch agent calls
- `incident_types.py` - Shared data structures
- `pyproject.toml` - Project dependencies
- `dispatch-local` - Local development runner

## Usage

### Development
```bash
cd /Users/matthew.pillari/go/src/github.com/DataDog/dispatch_agents/examples/dd_incident_langgraph_agent
./dispatch-local
```

### As Dispatch Agent
```python
import dispatch_agents

# Trigger investigation
result = await dispatch_agents.invoke(
    agent_name="dd-incident-agent",
    function_name="investigate",
    payload={"incident_id": "12345"},
    timeout=300,  # Investigations may take several minutes
)
print(result)  # Returns markdown investigation report
```

## Dependencies

- **LangGraph**: State-based workflow orchestration
- **LangChain**: LLM integration and tool framework
- **dispatch_agents**: Inter-agent communication
- **dd_mcp_agent**: Must be running to handle `dd_query` topic

## Key Features

- **Thread Safety**: Each investigation uses unique thread IDs to prevent conversation mixing
- **Error Handling**: Graceful degradation with informative error reports
- **Scalability**: Parallel hypothesis investigation for faster results
- **Flexibility**: Easy to modify workflow or add new investigation steps
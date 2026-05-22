# knowledge-base-query

Answers questions by searching a Confluence knowledge base. Builds a table of
contents from hub pages and their children, then uses Claude to fetch and
synthesize relevant pages into a structured answer.

## What This Demonstrates

| Handler | Decorator | Pattern |
|---------|-----------|---------|
| `query_knowledge_base` | `@fn()` | LLM-powered document retrieval and synthesis |

## Key Pattern: LLM Tool Use for Document Retrieval

The agent uses Claude Agent SDK to autonomously call Confluence APIs as tools,
decide which pages are relevant, and synthesize a final answer:

```python
from claude_agent_sdk import ClaudeAgentOptions, query

@fn()
async def query_knowledge_base(payload: KBQueryRequest) -> KBQueryResponse:
    options = ClaudeAgentOptions(tools=[fetch_page_tool, search_tool])
    async for event in query(payload.question, options=options):
        if isinstance(event, ResultMessage):
            return KBQueryResponse(answer=event.content)
```

## Setup

Configure your Confluence connection in the environment or `dispatch.yaml`:

```yaml
vars:
  confluence_base_url:
    value: "https://your-org.atlassian.net/wiki"
    description: "Confluence instance base URL"
```

Update `KNOWLEDGE_BASE_PAGES` and `SEARCHABLE_SPACES` in `agent.py` with your
Confluence hub page IDs and space keys before deploying.

## How to Run

```bash
cd examples/knowledge-base-query
dispatch agent deploy

dispatch function invoke --agent knowledge-base-query \
  --function query_knowledge_base \
  --payload '{"question": "How do I set up a new service?"}'
```

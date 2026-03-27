# Company Researcher

A Dispatch agent that generates a quick research report on any company using web search.

## What it does

Given a company name, the agent searches the web to compile a structured profile including:

- Company overview and core products/services
- Founding date, headquarters, and employee count
- Leadership team
- Funding history or public market status
- Competitors
- Recent news

## Usage

### Invoke directly

```python
from dispatch_agents import invoke

result = await invoke(
    agent_name="company-researcher",
    function_name="research_company",
    payload={"company": "Stripe"},
)
```

### Local development

```bash
cp .env.example .env  # No secrets needed - uses web search only
uv sync
dispatch agent dev
```

Then invoke via the MCP tools or CLI:

```bash
dispatch agent invoke company-researcher research_company '{"company": "Datadog"}'
```

## Response format

```json
{
  "company_name": "Datadog",
  "website": "https://www.datadoghq.com",
  "summary": "Cloud monitoring and security platform for developers and IT operations.",
  "founded": "2010",
  "headquarters": "New York, NY",
  "employee_count": "~5,000",
  "funding": "Public company (NASDAQ: DDOG)",
  "leadership": [
    {"name": "Olivier Pomel", "title": "CEO"},
    {"name": "Alexis Lê-Quôc", "title": "CTO"}
  ],
  "products_and_services": ["Infrastructure Monitoring", "APM", "Log Management", "Security"],
  "target_market": "Engineering and DevOps teams at cloud-native companies",
  "competitors": ["Splunk", "New Relic", "Dynatrace"],
  "recent_news": ["..."],
  "notes": ""
}
```

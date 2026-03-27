## Features

- **Dependency cooldowns**: Added cooldown support for `npm` and `uv` dependency managers, reducing redundant installs during agent execution.
- **New examples**: Added several new example agents to help you get started:
  - `company-researcher` – researches companies using available tools
  - `conversational-agent` – a basic interactive conversational agent
  - `daily-digest` – generates a daily digest of information
  - `deep-research` – multi-step deep research agent with configurable state and tools
  - `knowledge-base-query` – queries a knowledge base with configurable dispatch settings
  - `multi-framework` – demonstrates using multiple agent frameworks together
- **Example templates onboarding**: Improved onboarding flow for example templates.

## Bug Fixes

- Fixed a broken feedback link.

## Other Changes

- Renamed `.dispatch.yaml` files to `dispatch.yaml` across all examples for consistency.
- Removed the deprecated `dd_mcp_agent` example (superseded by `datadog_mcp` and `company-researcher`).
- Added a GitHub workflow lock/version check to enforce dependency hygiene.

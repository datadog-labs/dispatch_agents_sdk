## Features

- **Dependency cooldowns**: Added cooldown support for `npm` and `uv` dependency managers, reducing redundant install operations.
- **New examples**: Added several new example agents to the SDK:
  - `company-researcher` – researches companies using available tools
  - `conversational-agent` – a basic conversational agent template
  - `daily-digest` – generates daily digest summaries
  - `deep-research` – multi-step deep research agent with configurable state and tooling
  - `knowledge-base-query` – queries a knowledge base with configurable dispatch settings
  - `multi-framework` – demonstrates using multiple frameworks together
- **Templates onboarding**: Introduced template-based onboarding to help new users get started faster.

## Bug Fixes

- Fixed a broken feedback link.

## Other Changes

- Renamed `.dispatch.yaml` to `dispatch.yaml` across all examples for consistency.
- Removed the deprecated `dd_mcp_agent` example (superseded by `company-researcher` and `datadog_mcp`).
- Added a GitHub workflow for lock/version checks to keep dependencies in sync.

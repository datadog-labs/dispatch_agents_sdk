## Features

- **Dependency cooldowns**: Added cooldown support for npm and uv dependency management, reducing redundant installs during agent runs.
- **New examples**: Added several new example agents to help you get started:
  - `company-researcher` – researches companies using available tools
  - `conversational-agent` – a basic conversational agent template
  - `daily-digest` – generates periodic digest reports
  - `deep-research` – multi-step deep research agent with configurable state and tools
  - `knowledge-base-query` – queries a knowledge base with configurable dispatch settings
  - `multi-framework` – demonstrates integrating multiple agent frameworks together

## Bug Fixes

- Fixed a broken feedback link.

## Other Changes

- Renamed example dispatch config files from `.dispatch.yaml` to `dispatch.yaml` for consistency across all examples.
- Replaced the `dd_mcp_agent` example with the improved `company-researcher` example.
- Added GitHub workflow version and lock file checks to enforce dependency hygiene.
- Added onboarding templates to streamline new agent project setup.

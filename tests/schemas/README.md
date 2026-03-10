# GitHub Webhooks Schema

This directory contains the official GitHub webhooks JSON schema from [octokit/webhooks](https://github.com/octokit/webhooks), used to validate our SDK's GitHub event types.

## Updating the Schema

When GitHub releases new webhook types or modifies existing ones, update the schema:

```bash
curl -L -o sdk/tests/schemas/octokit-webhooks.json \
  https://unpkg.com/@octokit/webhooks-schemas/schema.json
```

Then run the compliance tests to see which new types need to be added:

```bash
cd sdk
uv run pytest tests/test_github_schema_compliance.py -v
```

The tests will:
- **FAIL** if SDK classes are missing required fields from the schema
- **WARN** about schema events not yet implemented in the SDK

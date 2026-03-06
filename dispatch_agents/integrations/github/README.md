# GitHub Integration

The SDK includes typed payloads for GitHub webhook events. See `__init__.py` for available event types.

## Schema Compliance Testing

The GitHub event types are verified against the official [octokit/webhooks](https://github.com/octokit/webhooks) JSON Schema. The schema is version-controlled at `tests/schemas/octokit-webhooks.json`.

To update the schema when GitHub releases new webhook types, see [tests/schemas/README.md](../../../tests/schemas/README.md).

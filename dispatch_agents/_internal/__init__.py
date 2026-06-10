"""Internal implementation packages for Dispatch Agents.

Anything under this package is private to the SDK implementation and carries no
backwards-compatibility guarantee. The public surface is the top-level
``dispatch_agents`` package and its documented modules.

Import policy:

- First-party packages in this monorepo (``backend``, ``cli``) MAY import from
  ``dispatch_agents._internal``. They version-lock with the SDK and share its
  internal wire/orchestration contracts; this coupling is deliberate, not a leak.
- External agent code MUST NOT import from ``_internal``. Use the public surface
  (``dispatch_agents`` and ``dispatch_agents.models``). The ``examples/`` agents
  serve as the proxy for external usage and are guarded by a test that forbids
  any ``_internal`` import there.
"""

# Lessons

- When adding CI logic that depends on remote git state, fail closed on fetch/discovery errors instead of silently continuing with stale local state.
- Release workflows need explicit serialization and least-privilege permissions; do not leave release jobs unconstrained or give write scopes to unrelated jobs.

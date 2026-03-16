# Workflow Refactor

## Checklist
- [x] Inspect current workflow layout and repo state.
- [x] Add reusable version-policy workflow.
- [x] Update feature-branch and release workflows to call the reusable version-policy workflow.
- [x] Move PyPI publish into `release.yml`.
- [x] Remove standalone `publish.yml`.
- [x] Run targeted verification.

## Review
- Added `.github/workflows/version-policy-reusable.yml` and routed both entry workflows through it.
- Moved PyPI publish into `release.yml` as a separate gated job that depends on successful version policy, CI, and release jobs.
- Removed `.github/workflows/publish.yml`, eliminating the prior `release.published` trigger path.
- Verification:
  - `uv run pytest .github/tests/test_version_policy.py -q` passed.
  - Manual workflow audit confirmed `current_tag` and `should_release` still flow through `needs.version_policy.outputs.*`.
  - Local YAML parse was not run because `PyYAML` is not installed in the base environment.

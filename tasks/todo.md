# CI Release Pipeline Follow-Ups

- [x] Add a reusable GitHub Actions CI workflow to remove drift between feature-branch and release pipelines.
- [x] Add a version-policy helper that enforces semantic version bumps for shipped SDK changes.
- [x] Add unit tests covering version-policy path classification and version comparison behavior.
- [x] Update `feature-branch.yml` and `release.yml` to use the helper and reusable workflow.
- [x] Verify the helper tests and workflow syntax, then record results below.

## Review

- Added `.github/workflows/ci-reusable.yml` and converted the feature-branch and release workflows into thin wrappers around it.
- Added `scripts/ci/version_policy.py` to classify changed paths, compare `pyproject.toml` against the latest semver tag, fail closed on tag fetch errors, and conservatively require bumps for unknown paths.
- Tightened the release workflows with serialized execution, least-privilege permissions, and commit-SHA-pinned third-party actions.
- Added `tests/test_version_policy.py` with 9 focused cases covering shipped code changes, docs-only changes, relevant `pyproject.toml` changes, tooling-only changes, downgraded versions, no-tag baselines, unknown paths, and fetch failure handling.
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_version_policy.py`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts/ci/version_policy.py tests/test_version_policy.py`
  - `ruby -e 'require "yaml"; Dir[".github/workflows/*.yml"].each { |path| YAML.load_file(path) }; puts "workflow yaml ok"'`

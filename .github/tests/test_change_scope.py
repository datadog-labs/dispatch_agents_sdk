"""Tests for the CI change scope helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "change_scope.py"
MODULE_NAME = "dispatch_agents_ci_change_scope"
sys.path.insert(0, str(MODULE_PATH.parent))

IGNORED_PATHS = {
    "README.md",
    "CONTRIBUTING.md",
    "NOTICE",
    "uv.lock",
    "pyproject.toml",
}
IGNORED_PREFIXES = (".github/", "tests/", "examples/", "plugins/", "LICENSE")

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec is not None
assert spec.loader is not None
change_scope = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = change_scope
spec.loader.exec_module(change_scope)


def test_docs_and_workflow_changes_are_not_release_relevant():
    assert (
        change_scope.classify_changed_files(
            [
                "README.md",
                ".github/workflows/feature-branch.yml",
                "tests/test_config.py",
            ],
            ignored_paths=set(IGNORED_PATHS),
            ignored_prefixes=IGNORED_PREFIXES,
        )
        is False
    )


def test_pyproject_change_is_deferred_to_semantic_diff():
    assert (
        change_scope.classify_changed_files(
            ["pyproject.toml"],
            ignored_paths=set(IGNORED_PATHS),
            ignored_prefixes=IGNORED_PREFIXES,
        )
        is False
    )


def test_sdk_change_is_release_relevant():
    assert (
        change_scope.classify_changed_files(
            ["dispatch_agents/instrument.py"],
            ignored_paths=set(IGNORED_PATHS),
            ignored_prefixes=IGNORED_PREFIXES,
        )
        is True
    )


def test_unknown_top_level_path_is_conservatively_relevant():
    assert (
        change_scope.classify_changed_files(
            ["new_surface/config.json"],
            ignored_paths=set(IGNORED_PATHS),
            ignored_prefixes=IGNORED_PREFIXES,
        )
        is True
    )


def test_custom_ignored_prefixes_make_policy_explicit():
    assert (
        change_scope.classify_changed_files(
            ["docs/release-notes.md"],
            ignored_paths=set(),
            ignored_prefixes=("docs/",),
        )
        is False
    )


def test_feature_branch_scope_uses_branch_base_for_pyproject():
    result = change_scope.determine_change_scope(
        mode="feature-branch",
        changed_files=[".github/workflows/release.yml"],
        latest_tag="v0.7.3",
        feature_branch_base_ref="abc123",
        ignored_paths=set(IGNORED_PATHS),
        ignored_prefixes=IGNORED_PREFIXES,
    )

    assert result.range_label == "abc123...HEAD"
    assert result.pyproject_baseline_ref == "abc123"
    assert result.source_changed is False


def test_release_scope_uses_latest_tag():
    result = change_scope.determine_change_scope(
        mode="release",
        changed_files=["dispatch_agents/instrument.py"],
        latest_tag="v0.7.3",
        feature_branch_base_ref=None,
        ignored_paths=set(IGNORED_PATHS),
        ignored_prefixes=IGNORED_PREFIXES,
    )

    assert result.range_label == "v0.7.3...HEAD"
    assert result.pyproject_baseline_ref == "v0.7.3"
    assert result.source_changed is True


def test_feature_branch_scope_requires_base_ref():
    with pytest.raises(ValueError):
        change_scope.determine_change_scope(
            mode="feature-branch",
            changed_files=[],
            latest_tag="v0.7.3",
            feature_branch_base_ref=None,
            ignored_paths=set(IGNORED_PATHS),
            ignored_prefixes=IGNORED_PREFIXES,
        )


def test_parse_json_string_list_rejects_non_string_lists():
    with pytest.raises(ValueError):
        change_scope.parse_json_string_list('["ok", 1]')


def test_parse_args_requires_explicit_policy_values(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "change_scope.py",
            "--mode",
            "release",
            "--ignored-paths-json",
            '["README.md"]',
            "--ignored-prefixes-json",
            '[".github/"]',
        ],
    )

    args = change_scope.parse_args()

    assert args.ignored_paths_json == '["README.md"]'
    assert args.ignored_prefixes_json == '[".github/"]'

"""Tests for the CI version policy helper."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "version_policy.py"
)
MODULE_NAME = "dispatch_agents_ci_version_policy"
sys.path.insert(0, str(MODULE_PATH.parent))

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec is not None
assert spec.loader is not None
version_policy = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = version_policy
spec.loader.exec_module(version_policy)


def pyproject(
    *,
    version: str,
    description: str = "Dispatch Agents SDK",
    build_requires: list[str] | None = None,
    hatch_packages: list[str] | None = None,
    dev_dependencies: list[str] | None = None,
) -> dict[str, object]:
    return {
        "build-system": {
            "requires": build_requires or ["hatchling"],
            "build-backend": "hatchling.build",
        },
        "project": {
            "name": "dispatch_agents",
            "version": version,
            "description": description,
        },
        "tool": {
            "hatch": {
                "build": {
                    "targets": {
                        "wheel": {
                            "packages": hatch_packages
                            or ["dispatch_agents", "agentservice"]
                        }
                    }
                }
            },
            "ruff": {"line-length": 88},
        },
        "dependency-groups": {
            "dev": dev_dependencies or ["pytest>=8.4.2"],
        },
    }


def test_sdk_change_requires_bump_when_version_is_unchanged():
    result = version_policy.evaluate_policy(
        source_changed=True,
        current_pyproject=pyproject(version="0.7.3"),
        baseline_pyproject=pyproject(version="0.7.2"),
        latest_tag="v0.7.3",
    )

    assert result.requires_version_bump is True
    assert result.has_version_bump is False
    assert result.failure_reason is not None


def test_sdk_change_passes_with_higher_version():
    result = version_policy.evaluate_policy(
        source_changed=True,
        current_pyproject=pyproject(version="0.7.4"),
        baseline_pyproject=pyproject(version="0.7.3"),
        latest_tag="v0.7.3",
    )

    assert result.requires_version_bump is True
    assert result.has_version_bump is True
    assert result.should_release is True
    assert result.failure_reason is None


def test_docs_only_change_does_not_require_bump():
    result = version_policy.evaluate_policy(
        source_changed=False,
        current_pyproject=pyproject(version="0.7.3"),
        baseline_pyproject=pyproject(version="0.7.3"),
        latest_tag="v0.7.3",
    )

    assert result.requires_version_bump is False
    assert result.has_version_bump is False
    assert result.should_release is False
    assert result.failure_reason is None


def test_relevant_pyproject_change_requires_bump():
    result = version_policy.evaluate_policy(
        source_changed=False,
        current_pyproject=pyproject(version="0.7.3", description="Updated SDK"),
        baseline_pyproject=pyproject(version="0.7.2"),
        latest_tag="v0.7.3",
    )

    assert result.relevant_pyproject_changed is True
    assert result.requires_version_bump is True
    assert result.failure_reason is not None


def test_dev_tooling_pyproject_change_does_not_require_bump():
    result = version_policy.evaluate_policy(
        source_changed=False,
        current_pyproject=pyproject(
            version="0.7.3",
            dev_dependencies=["pytest>=8.4.2", "ruff>=0.14.1"],
        ),
        baseline_pyproject=pyproject(version="0.7.3"),
        latest_tag="v0.7.3",
    )

    assert result.relevant_pyproject_changed is False
    assert result.requires_version_bump is False
    assert result.failure_reason is None


def test_lower_than_latest_release_fails_even_for_docs_change():
    result = version_policy.evaluate_policy(
        source_changed=False,
        current_pyproject=pyproject(version="0.7.2"),
        baseline_pyproject=pyproject(version="0.7.3"),
        latest_tag="v0.7.3",
    )

    assert result.failure_reason == (
        "Current version v0.7.2 is behind latest release v0.7.3."
    )


def test_no_tag_baseline_allows_initial_version():
    result = version_policy.evaluate_policy(
        source_changed=True,
        current_pyproject=pyproject(version="0.1.0"),
        baseline_pyproject=None,
        latest_tag=None,
    )

    assert result.latest_tag == "v0.0.0"
    assert result.requires_version_bump is True
    assert result.has_version_bump is True
    assert result.failure_reason is None


def test_workflow_classified_release_relevant_change_requires_bump():
    result = version_policy.evaluate_policy(
        source_changed=True,
        current_pyproject=pyproject(version="0.7.3"),
        baseline_pyproject=pyproject(version="0.7.2"),
        latest_tag="v0.7.3",
    )

    assert result.requires_version_bump is True
    assert result.failure_reason is not None


def test_fetch_tags_raises_on_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["git", "fetch", "--force", "--tags", "origin"],
        )

    monkeypatch.setattr(version_policy, "fetch_tags", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        version_policy.fetch_tags()

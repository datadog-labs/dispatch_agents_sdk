"""Tests for shared CI git helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "ci_git.py"
MODULE_NAME = "dispatch_agents_ci_git"
sys.path.insert(0, str(MODULE_PATH.parent))

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec is not None
assert spec.loader is not None
ci_git = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = ci_git
spec.loader.exec_module(ci_git)


def test_parse_tag_parses_semver():
    assert ci_git.parse_tag("v1.2.3") == (1, 2, 3)


def test_parse_tag_rejects_invalid_tag():
    with pytest.raises(ValueError):
        ci_git.parse_tag("release-1.2.3")


def test_filter_semver_tags_filters_and_sorts():
    assert ci_git.filter_semver_tags(["v1.10.0", "foo", "v1.2.0"]) == [
        "v1.2.0",
        "v1.10.0",
    ]


def test_get_latest_tag_returns_none_when_no_semver_tags(monkeypatch):
    monkeypatch.setattr(ci_git, "run_git", lambda *args, **kwargs: "foo\nbar")

    assert ci_git.get_latest_tag() is None


def test_get_latest_tag_returns_last_semver(monkeypatch):
    monkeypatch.setattr(ci_git, "run_git", lambda *args, **kwargs: "v0.1.0\nv0.2.0\n")

    assert ci_git.get_latest_tag() == "v0.2.0"


def test_get_merge_base_delegates_to_git(monkeypatch):
    monkeypatch.setattr(ci_git, "run_git", lambda *args, **kwargs: "abc123")

    assert ci_git.get_merge_base("HEAD", "origin/main") == "abc123"


def test_get_changed_files_for_ref(monkeypatch):
    monkeypatch.setattr(
        ci_git,
        "run_git",
        lambda *args, **kwargs: "dispatch_agents/foo.py\nREADME.md\n",
    )

    assert ci_git.get_changed_files("v0.1.0") == [
        "dispatch_agents/foo.py",
        "README.md",
    ]


def test_get_changed_files_without_ref_uses_ls_files(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run_git(*args, **kwargs):
        calls.append(args)
        return "a.py\n"

    monkeypatch.setattr(ci_git, "run_git", fake_run_git)

    assert ci_git.get_changed_files(None) == ["a.py"]
    assert calls == [("ls-files",)]


def test_fetch_tags_raises_on_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["git", "fetch", "--force", "--tags", "origin"],
        )

    monkeypatch.setattr(ci_git.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        ci_git.fetch_tags()


def test_fetch_main_branch_ref_raises_on_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=[
                "git",
                "fetch",
                "--no-tags",
                "origin",
                "main:refs/remotes/origin/main",
            ],
        )

    monkeypatch.setattr(ci_git.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        ci_git.fetch_main_branch_ref()

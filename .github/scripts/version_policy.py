#!/usr/bin/env python3
"""Enforce release version policy for CI workflows."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
SEMVER_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class PolicyResult:
    current_version: str
    current_tag: str
    latest_tag: str
    source_changed: bool
    requires_version_bump: bool
    has_version_bump: bool
    should_release: bool
    relevant_pyproject_changed: bool
    failure_reason: str | None


def parse_version(version: str) -> tuple[int, int, int]:
    match = SEMVER_VERSION_RE.fullmatch(version)
    if not match:
        raise ValueError(f"Unsupported version format: {version}")
    return tuple(int(part) for part in match.groups())


def parse_tag(tag: str) -> tuple[int, int, int]:
    match = SEMVER_TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(f"Unsupported tag format: {tag}")
    return tuple(int(part) for part in match.groups())


def filter_semver_tags(tags: Iterable[str]) -> list[str]:
    valid_tags = [tag for tag in tags if SEMVER_TAG_RE.fullmatch(tag)]
    return sorted(valid_tags, key=parse_tag)


def compare_versions(current_version: str, latest_tag: str) -> int:
    current = parse_version(current_version)
    latest = parse_tag(latest_tag)
    if current > latest:
        return 1
    if current < latest:
        return -1
    return 0


def is_relevant_pyproject_change(
    current_pyproject: dict[str, Any], baseline_pyproject: dict[str, Any] | None
) -> bool:
    if baseline_pyproject is None:
        return False

    current_project = dict(current_pyproject.get("project", {}))
    baseline_project = dict(baseline_pyproject.get("project", {}))
    current_project.pop("version", None)
    baseline_project.pop("version", None)

    current_hatch = current_pyproject.get("tool", {}).get("hatch", {}).get("build", {})
    baseline_hatch = (
        baseline_pyproject.get("tool", {}).get("hatch", {}).get("build", {})
    )

    return (
        current_pyproject.get("build-system", {})
        != baseline_pyproject.get("build-system", {})
        or current_project != baseline_project
        or current_hatch != baseline_hatch
    )


def evaluate_policy(
    *,
    source_changed: bool,
    current_pyproject: dict[str, Any],
    baseline_pyproject: dict[str, Any] | None,
    latest_tag: str | None,
) -> PolicyResult:
    current_version = current_pyproject["project"]["version"]
    current_tag = f"v{current_version}"
    baseline_tag = latest_tag or "v0.0.0"
    relevant_change = is_relevant_pyproject_change(
        current_pyproject, baseline_pyproject
    )
    bump_required = source_changed or relevant_change

    has_bump = compare_versions(current_version, baseline_tag) > 0
    comparison = compare_versions(current_version, baseline_tag)
    failure_reason: str | None = None

    if comparison < 0:
        failure_reason = (
            f"Current version {current_tag} is behind latest release {baseline_tag}."
        )
    elif latest_tag is not None and bump_required and comparison <= 0:
        failure_reason = (
            "Changes require a semantic version bump, "
            f"but {current_tag} is not greater than {baseline_tag}."
        )

    return PolicyResult(
        current_version=current_version,
        current_tag=current_tag,
        latest_tag=baseline_tag,
        source_changed=source_changed,
        requires_version_bump=bump_required,
        has_version_bump=has_bump,
        should_release=has_bump,
        relevant_pyproject_changed=relevant_change,
        failure_reason=failure_reason,
    )


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def fetch_tags() -> None:
    subprocess.run(
        ["git", "fetch", "--force", "--tags", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )


def get_latest_tag() -> str | None:
    tags_output = run_git("tag", "--list", "v*")
    tags = [line.strip() for line in tags_output.splitlines() if line.strip()]
    valid_tags = filter_semver_tags(tags)
    if not valid_tags:
        return None
    return valid_tags[-1]


def load_pyproject(path: Path) -> dict[str, Any]:
    with path.open("rb") as file_obj:
        return tomllib.load(file_obj)


def load_pyproject_from_ref(ref: str | None, repo_root: Path) -> dict[str, Any] | None:
    if ref is None:
        return None

    result = subprocess.run(
        ["git", "show", f"{ref}:pyproject.toml"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        return None
    return tomllib.loads(result.stdout)


def write_github_outputs(result: PolicyResult) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    outputs = {
        "current_version": result.current_version,
        "current_tag": result.current_tag,
        "latest_tag": result.latest_tag,
        "requires_version_bump": str(result.requires_version_bump).lower(),
        "has_version_bump": str(result.has_version_bump).lower(),
        "relevant_pyproject_changed": str(result.relevant_pyproject_changed).lower(),
        "should_release": str(result.should_release).lower(),
    }

    with Path(output_path).open("a", encoding="utf-8") as file_obj:
        for key, value in outputs.items():
            file_obj.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("feature-branch", "release"),
        required=True,
        help="Execution mode for workflow messaging.",
    )
    parser.add_argument(
        "--source-changed",
        choices=("true", "false"),
        required=True,
        help="Whether the workflow determined that release-relevant non-pyproject files changed.",
    )
    parser.add_argument(
        "--pyproject-baseline-ref",
        default="",
        help="Git ref to use as the pyproject.toml comparison baseline. Defaults to the latest release tag when omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    fetch_tags()

    latest_tag = get_latest_tag()
    current_pyproject = load_pyproject(repo_root / "pyproject.toml")
    pyproject_baseline_ref = args.pyproject_baseline_ref or latest_tag
    baseline_pyproject = load_pyproject_from_ref(pyproject_baseline_ref, repo_root)
    source_changed = args.source_changed == "true"

    result = evaluate_policy(
        source_changed=source_changed,
        current_pyproject=current_pyproject,
        baseline_pyproject=baseline_pyproject,
        latest_tag=latest_tag,
    )
    write_github_outputs(result)

    mode_label = "feature branch" if args.mode == "feature-branch" else "release"
    print(f"Version policy check ({mode_label})")
    print(f"  latest tag: {result.latest_tag}")
    print(f"  current tag: {result.current_tag}")
    print(f"  pyproject baseline: {pyproject_baseline_ref or '(none)'}")
    print(f"  source changed: {str(result.source_changed).lower()}")
    print(f"  requires bump: {str(result.requires_version_bump).lower()}")
    print(f"  should release: {str(result.should_release).lower()}")

    if result.failure_reason:
        print(result.failure_reason, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

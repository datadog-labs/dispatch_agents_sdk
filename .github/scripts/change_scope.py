#!/usr/bin/env python3
"""Classify whether the current ref includes release-relevant changes."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
IGNORED_PATHS = {
    "README.md",
    "CONTRIBUTING.md",
    "NOTICE",
    "uv.lock",
    "pyproject.toml",
}
IGNORED_PREFIXES = (".github/", "tests/", "examples/", "plugins/")


@dataclass(frozen=True)
class ChangeScopeResult:
    mode: str
    range_label: str
    pyproject_baseline_ref: str | None
    changed_files: tuple[str, ...]
    source_changed: bool


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def filter_semver_tags(tags: list[str]) -> list[str]:
    valid_tags = [tag for tag in tags if SEMVER_TAG_RE.fullmatch(tag)]
    return sorted(valid_tags, key=lambda tag: tuple(map(int, tag[1:].split("."))))


def get_latest_tag() -> str | None:
    tags_output = run_git("tag", "--list", "v*")
    tags = [line.strip() for line in tags_output.splitlines() if line.strip()]
    valid_tags = filter_semver_tags(tags)
    if not valid_tags:
        return None
    return valid_tags[-1]


def is_release_relevant_source_path(path: str) -> bool:
    if path in IGNORED_PATHS:
        return False
    if path.startswith("LICENSE"):
        return False
    return not path.startswith(IGNORED_PREFIXES)


def classify_changed_files(changed_files: list[str]) -> bool:
    return any(is_release_relevant_source_path(path) for path in changed_files)


def determine_change_scope(
    *,
    mode: str,
    changed_files: list[str],
    latest_tag: str | None,
    feature_branch_base_ref: str | None,
) -> ChangeScopeResult:
    if mode == "feature-branch":
        if feature_branch_base_ref is None:
            raise ValueError("feature_branch_base_ref is required for feature-branch mode")
        return ChangeScopeResult(
            mode=mode,
            range_label=f"{feature_branch_base_ref}...HEAD",
            pyproject_baseline_ref=feature_branch_base_ref,
            changed_files=tuple(changed_files),
            source_changed=classify_changed_files(changed_files),
        )

    range_label = f"{latest_tag}...HEAD" if latest_tag is not None else "tracked files in HEAD"
    return ChangeScopeResult(
        mode=mode,
        range_label=range_label,
        pyproject_baseline_ref=latest_tag,
        changed_files=tuple(changed_files),
        source_changed=classify_changed_files(changed_files),
    )


def get_feature_branch_base_ref() -> str:
    subprocess.run(
        ["git", "fetch", "--no-tags", "origin", "main:refs/remotes/origin/main"],
        check=True,
        capture_output=True,
        text=True,
    )
    return run_git("merge-base", "HEAD", "origin/main")


def get_changed_files(diff_ref: str | None) -> list[str]:
    if diff_ref is None:
        output = run_git("ls-files")
    else:
        output = run_git(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{diff_ref}...HEAD",
        )
    return [line.strip() for line in output.splitlines() if line.strip()]


def write_github_outputs(result: ChangeScopeResult) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    outputs = {
        "source_changed": str(result.source_changed).lower(),
        "pyproject_baseline_ref": result.pyproject_baseline_ref or "",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subprocess.run(
        ["git", "fetch", "--force", "--tags", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )

    latest_tag = get_latest_tag()
    if args.mode == "feature-branch":
        feature_branch_base_ref = get_feature_branch_base_ref()
        changed_files = get_changed_files(feature_branch_base_ref)
    else:
        feature_branch_base_ref = None
        changed_files = get_changed_files(latest_tag)

    result = determine_change_scope(
        mode=args.mode,
        changed_files=changed_files,
        latest_tag=latest_tag,
        feature_branch_base_ref=feature_branch_base_ref,
    )
    write_github_outputs(result)

    print(f"Change scope ({args.mode})")
    print(f"  range: {result.range_label}")
    print(
        f"  pyproject baseline: {result.pyproject_baseline_ref or '(latest tag baseline unavailable)'}"
    )
    print(f"  release-relevant source changed: {str(result.source_changed).lower()}")
    if result.changed_files:
        print("  compared files:")
        for path in result.changed_files:
            print(f"    - {path}")
    else:
        print("  compared files: (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

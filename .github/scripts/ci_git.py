#!/usr/bin/env python3
"""Shared git and tag helpers for CI workflow scripts."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_tag(tag: str) -> tuple[int, int, int]:
    match = SEMVER_TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(f"Unsupported tag format: {tag}")
    return tuple(int(part) for part in match.groups())


def filter_semver_tags(tags: Iterable[str]) -> list[str]:
    valid_tags = [tag for tag in tags if SEMVER_TAG_RE.fullmatch(tag)]
    return sorted(valid_tags, key=parse_tag)


def fetch_tags() -> None:
    subprocess.run(
        ["git", "fetch", "--force", "--tags", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )


def fetch_main_branch_ref() -> None:
    subprocess.run(
        ["git", "fetch", "--no-tags", "origin", "main:refs/remotes/origin/main"],
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


def get_merge_base(left_ref: str, right_ref: str) -> str:
    return run_git("merge-base", left_ref, right_ref)


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

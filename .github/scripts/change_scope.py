#!/usr/bin/env python3
"""Classify whether the current ref includes release-relevant changes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ci_git import (
    fetch_main_branch_ref,
    fetch_tags,
    get_changed_files,
    get_latest_tag,
    get_merge_base,
)


@dataclass(frozen=True)
class ChangeScopeResult:
    ref_name: str
    range_label: str
    pyproject_baseline_ref: str | None
    changed_files: tuple[str, ...]
    source_changed: bool


def parse_json_string_list(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError("expected a JSON array of strings")
    return parsed


def is_release_relevant_source_path(
    path: str,
    *,
    ignored_paths: set[str],
    ignored_prefixes: tuple[str, ...],
) -> bool:
    if path in ignored_paths:
        return False
    return not path.startswith(ignored_prefixes)


def classify_changed_files(
    changed_files: list[str],
    *,
    ignored_paths: set[str],
    ignored_prefixes: tuple[str, ...],
) -> bool:
    return any(
        is_release_relevant_source_path(
            path,
            ignored_paths=ignored_paths,
            ignored_prefixes=ignored_prefixes,
        )
        for path in changed_files
    )


def determine_change_scope(
    *,
    ref_name: str,
    changed_files: list[str],
    latest_tag: str | None,
    feature_branch_base_ref: str | None,
    ignored_paths: set[str],
    ignored_prefixes: tuple[str, ...],
) -> ChangeScopeResult:
    if ref_name != "main":
        if feature_branch_base_ref is None:
            raise ValueError(
                "feature_branch_base_ref is required for feature-branch mode"
            )
        return ChangeScopeResult(
            ref_name=ref_name,
            range_label=f"{feature_branch_base_ref}...HEAD",
            pyproject_baseline_ref=feature_branch_base_ref,
            changed_files=tuple(changed_files),
            source_changed=classify_changed_files(
                changed_files,
                ignored_paths=ignored_paths,
                ignored_prefixes=ignored_prefixes,
            ),
        )

    range_label = (
        f"{latest_tag}...HEAD" if latest_tag is not None else "tracked files in HEAD"
    )
    return ChangeScopeResult(
        ref_name=ref_name,
        range_label=range_label,
        pyproject_baseline_ref=latest_tag,
        changed_files=tuple(changed_files),
        source_changed=classify_changed_files(
            changed_files,
            ignored_paths=ignored_paths,
            ignored_prefixes=ignored_prefixes,
        ),
    )


def get_feature_branch_base_ref() -> str:
    fetch_main_branch_ref()
    return get_merge_base("HEAD", "origin/main")


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
        "--ref-name",
        required=True,
        help="GitHub ref name for the current workflow run.",
    )
    parser.add_argument(
        "--ignored-paths-json",
        required=True,
        help="JSON array of exact paths that do not count as release-relevant source changes.",
    )
    parser.add_argument(
        "--ignored-prefixes-json",
        required=True,
        help="JSON array of path prefixes that do not count as release-relevant source changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ignored_paths = set(parse_json_string_list(args.ignored_paths_json))
    ignored_prefixes = tuple(parse_json_string_list(args.ignored_prefixes_json))

    fetch_tags()

    latest_tag = get_latest_tag()
    if args.ref_name != "main":
        feature_branch_base_ref = get_feature_branch_base_ref()
        changed_files = get_changed_files(feature_branch_base_ref)
    else:
        feature_branch_base_ref = None
        changed_files = get_changed_files(latest_tag)

    result = determine_change_scope(
        ref_name=args.ref_name,
        changed_files=changed_files,
        latest_tag=latest_tag,
        feature_branch_base_ref=feature_branch_base_ref,
        ignored_paths=ignored_paths,
        ignored_prefixes=ignored_prefixes,
    )
    write_github_outputs(result)

    print(f"Change scope ({args.ref_name})")
    print(f"  range: {result.range_label}")
    print(
        f"  pyproject baseline: {result.pyproject_baseline_ref or '(latest tag baseline unavailable)'}"
    )
    print(f"  ignored paths: {sorted(ignored_paths)}")
    print(f"  ignored prefixes: {list(ignored_prefixes)}")
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

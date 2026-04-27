"""Verify SDK GitHub types match the official octokit/webhooks schema.

This module uses Pydantic's built-in JSON schema generation to compare
our SDK types against the official GitHub webhook schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import dispatch_agents.integrations.github as gh_module

SCHEMA_PATH = Path(__file__).parent / "schemas" / "octokit-webhooks.json"

# Type mapping from Pydantic class names to octokit schema type names
PYDANTIC_TO_OCTOKIT_TYPE = {
    "GitHubUser": {
        "user",
        "simple-user",
        "nullable-simple-user",
        "organization",
        "github-org",
        "object",
    },
    "GitHubDependabotAlert": {"dependabot-alert"},
    "GitHubRepository": {"repository", "repository-lite"},
    "GitHubPullRequest": {
        "pull-request",
        "simple-pull-request",
        "pull-request-minimal",
        "object",
    },
    "GitHubInstallation": {"installation", "installation-lite"},
    "GitHubChanges": {"object"},
    "GitHubChangeValue": {"object"},
    "GitHubLink": {"object"},
    "GitHubIssuePullRequest": {"object"},
    "GitHubLabel": {"label"},
    "GitHubIssue": {"issue"},
    "GitHubComment": {"issue-comment"},
    "GitHubReviewComment": {"pull-request-review-comment"},
    "GitHubMilestone": {"milestone", "nullable-milestone"},
    "GitHubReview": {"pull-request-review"},
    "GitHubCheckRun": {"check-run", "object"},
    "GitHubCheckSuite": {"check-suite", "object"},
    "GitHubWorkflowRun": {"workflow-run", "deployment-workflow-run", "object"},
    "GitHubWorkflow": {"workflow", "object"},
    "GitHubDeployment": {"deployment"},
    "GitHubWorkflowJob": {"workflow-job", "object"},
    "GitHubRelease": {"release", "release-1"},
    "GitHubCommit": {"commit", "object"},
    "GitHubCommitUser": {"committer", "object"},
    "GitHubRequestedAction": {"requested-action", "object"},
    "GitHubReviewThread": {"object"},
    "GitHubDiscussion": {"discussion"},
    "GitHubSecretScanningAlert": {"secret-scanning-alert", "object"},
    "GitHubTeam": {"team"},
    "NoneType": {"null"},
}


@pytest.fixture(scope="module")
def octokit_schema() -> dict[str, Any]:
    """Load the version-controlled octokit webhooks schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def get_sdk_event_classes() -> dict[str, type[Any]]:
    """Get all SDK event classes mapped by their dispatch topic."""
    classes: dict[str, type[Any]] = {}
    for name in dir(gh_module):
        cls = getattr(gh_module, name)
        if (
            isinstance(cls, type)
            and hasattr(cls, "_dispatch_topic")
            and cls._dispatch_topic.startswith("github.")
        ):
            classes[cls._dispatch_topic] = cls
    return classes


def topic_to_octokit_def_name(topic: str) -> str:
    """Convert dispatch topic to octokit definition name."""
    event_name = topic.replace("github.", "", 1)
    if "." not in event_name:
        return f"{event_name}$event"
    return event_name.replace(".", "$")


def extract_pydantic_type(prop: dict[str, Any]) -> tuple[str, bool]:
    """Extract type name and nullability from Pydantic schema property."""
    if "$ref" in prop:
        return prop["$ref"].split("/")[-1], False

    if "anyOf" in prop:
        types, is_nullable = [], False
        for opt in prop["anyOf"]:
            if opt.get("type") == "null":
                is_nullable = True
            elif "$ref" in opt:
                types.append(opt["$ref"].split("/")[-1])
            elif "type" in opt:
                types.append(opt["type"])
        return types[0] if types else "unknown", is_nullable

    if "type" in prop:
        t = prop["type"]
        if isinstance(t, list):
            is_nullable = "null" in t
            non_null = [x for x in t if x != "null"]
            return non_null[0] if non_null else "null", is_nullable
        return t, False

    # Handle allOf (used for extended types)
    if "allOf" in prop:
        for opt in prop["allOf"]:
            if "$ref" in opt:
                return opt["$ref"].split("/")[-1], False
        return "object", False

    return "unknown", False


def extract_octokit_type(prop: dict[str, Any]) -> tuple[str, bool]:
    """Extract type name and nullability from octokit schema property."""
    if "$ref" in prop:
        return prop["$ref"].split("/")[-1], False

    if "anyOf" in prop or "oneOf" in prop:
        variants = prop.get("anyOf", prop.get("oneOf", []))
        types, is_nullable = [], False
        for opt in variants:
            if opt.get("type") == "null":
                is_nullable = True
            elif "$ref" in opt:
                types.append(opt["$ref"].split("/")[-1])
            elif "type" in opt:
                types.append(opt["type"])
        return types[0] if types else "unknown", is_nullable

    if "allOf" in prop:
        for opt in prop["allOf"]:
            if "$ref" in opt:
                return opt["$ref"].split("/")[-1], False
        return "object", False

    if "type" in prop:
        t = prop["type"]
        if isinstance(t, list):
            is_nullable = "null" in t
            non_null = [x for x in t if x != "null"]
            return non_null[0] if non_null else "null", is_nullable
        return t, False

    return "unknown", False


def types_are_compatible(pydantic_type: str, octokit_type: str) -> bool:
    """Check if Pydantic type is compatible with octokit type."""
    if pydantic_type == octokit_type:
        return True

    # Check type mapping
    if pydantic_type in PYDANTIC_TO_OCTOKIT_TYPE:
        return octokit_type in PYDANTIC_TO_OCTOKIT_TYPE[pydantic_type]

    # Generic object types are compatible
    if pydantic_type == "object" or octokit_type == "object":
        return True

    # dict in Pydantic = object in JSON Schema
    if pydantic_type == "dict" and octokit_type == "object":
        return True

    return False


def test_sdk_fields_exist_in_octokit_schema(octokit_schema: dict[str, Any]):
    """Verify all SDK fields exist in the official schema."""
    sdk_classes = get_sdk_event_classes()
    errors: list[str] = []

    for topic, sdk_class in sdk_classes.items():
        def_name = topic_to_octokit_def_name(topic)
        octokit_def = octokit_schema.get("definitions", {}).get(def_name)

        if not octokit_def or "properties" not in octokit_def:
            continue

        # Generate Pydantic schema and compare properties
        pydantic_schema = sdk_class.model_json_schema()
        pydantic_props = set(pydantic_schema.get("properties", {}).keys())
        octokit_props = set(octokit_def.get("properties", {}).keys())

        # Fields in SDK but not in octokit schema
        extra_fields = pydantic_props - octokit_props
        for field in extra_fields:
            # Check if field is nullable (optional) - those are OK to have extra
            prop = pydantic_schema["properties"][field]
            _, is_nullable = extract_pydantic_type(prop)
            if not is_nullable:
                errors.append(
                    f"{sdk_class.__name__}.{field}: required field not in schema"
                )

    if errors:
        pytest.fail(
            f"SDK has fields not in official schema ({len(errors)}):\n"
            + "\n".join(sorted(errors)[:20])
            + ("\n..." if len(errors) > 20 else "")
        )


def test_sdk_field_types_match_octokit(octokit_schema: dict[str, Any]):
    """Verify SDK field types match the official schema."""
    sdk_classes = get_sdk_event_classes()
    type_errors: list[str] = []
    nullability_errors: list[str] = []

    for topic, sdk_class in sdk_classes.items():
        def_name = topic_to_octokit_def_name(topic)
        octokit_def = octokit_schema.get("definitions", {}).get(def_name)

        if not octokit_def or "properties" not in octokit_def:
            continue

        pydantic_schema = sdk_class.model_json_schema()
        pydantic_props = pydantic_schema.get("properties", {})
        octokit_props = octokit_def.get("properties", {})
        octokit_required = set(octokit_def.get("required", []))

        # Compare common fields
        common_fields = set(pydantic_props.keys()) & set(octokit_props.keys())

        for field in common_fields:
            p_type, p_null = extract_pydantic_type(pydantic_props[field])
            o_type, o_null = extract_octokit_type(octokit_props[field])

            if not types_are_compatible(p_type, o_type):
                type_errors.append(
                    f"{sdk_class.__name__}.{field}: SDK '{p_type}' != schema '{o_type}'"
                )

            # Only check nullability for required fields.
            # For optional fields (not in required), SDK can use None to represent "absent".
            # Skip inherited nullable fields: base classes declare some fields as optional
            # to cover the broadest set of event shapes (e.g., repository is absent for
            # org-level events). Subclasses that don't override inherit the nullable type,
            # even though the specific event schema marks the field as required.
            field_is_inherited_nullable = (
                p_null and not o_null and field not in sdk_class.__annotations__
            )
            if (
                field in octokit_required
                and p_null != o_null
                and not field_is_inherited_nullable
            ):
                nullability_errors.append(
                    f"{sdk_class.__name__}.{field}: SDK null={p_null}, schema null={o_null}"
                )

    all_errors = type_errors + nullability_errors
    if all_errors:
        msg = ""
        if type_errors:
            msg += f"Type mismatches ({len(type_errors)}):\n"
            msg += "\n".join(sorted(type_errors)[:10])
            if len(type_errors) > 10:
                msg += "\n..."
        if nullability_errors:
            if msg:
                msg += "\n\n"
            msg += f"Nullability mismatches ({len(nullability_errors)}):\n"
            msg += "\n".join(sorted(nullability_errors)[:10])
            if len(nullability_errors) > 10:
                msg += "\n..."
        pytest.fail(msg)


def test_sdk_has_required_octokit_fields(octokit_schema: dict[str, Any]):
    """Verify SDK has all fields marked required in the schema."""
    sdk_classes = get_sdk_event_classes()
    errors: list[str] = []

    for topic, sdk_class in sdk_classes.items():
        def_name = topic_to_octokit_def_name(topic)
        octokit_def = octokit_schema.get("definitions", {}).get(def_name)

        if not octokit_def:
            continue

        octokit_required = set(octokit_def.get("required", []))
        if not octokit_required:
            continue

        pydantic_schema = sdk_class.model_json_schema()
        pydantic_props = set(pydantic_schema.get("properties", {}).keys())

        missing = octokit_required - pydantic_props
        if missing:
            errors.append(f"{sdk_class.__name__}: missing required fields {missing}")

    if errors:
        pytest.fail("SDK missing required fields:\n" + "\n".join(sorted(errors)))


def test_sdk_event_coverage(octokit_schema: dict[str, Any]):
    """Every octokit webhook event must have a corresponding SDK class.

    This test hard-fails if any event defined in the official octokit schema
    is not implemented in the SDK. When GitHub adds new events, update the
    schema (see sdk/tests/schemas/README.md) and then add the missing classes
    to dispatch_agents/integrations/github/__init__.py.
    """
    sdk_classes = get_sdk_event_classes()
    sdk_topics = set(sdk_classes.keys())

    # Extract all event definitions from octokit schema
    octokit_topics: set[str] = set()
    for def_name in octokit_schema.get("definitions", {}).keys():
        if "$" in def_name:  # Event definitions use $ as separator
            topic = "github." + def_name.replace("$", ".")
            # Handle event$event pattern (events without actions, e.g. gollum$event)
            if topic.endswith(".event"):
                topic = topic.rsplit(".event", 1)[0]
            octokit_topics.add(topic)

    missing = sorted(octokit_topics - sdk_topics)

    if missing:
        pytest.fail(
            f"SDK is missing {len(missing)} GitHub webhook event(s).\n"
            "Add the missing classes to "
            "dispatch_agents/integrations/github/__init__.py.\n"
            "Missing topics:\n" + "\n".join(f"  - {t}" for t in missing)
        )


@pytest.mark.parametrize(
    ("model_name", "definition_name", "field_name", "required", "nullable", "types"),
    [
        (
            "GitHubRepository",
            "repository",
            "created_at",
            True,
            False,
            {"integer", "string"},
        ),
        (
            "GitHubRepository",
            "repository",
            "pushed_at",
            True,
            True,
            {"integer", "string"},
        ),
        (
            "GitHubCommitUser",
            "committer",
            "email",
            True,
            True,
            {"string"},
        ),
    ],
)
def test_confirmed_shared_model_fields_match_octokit(
    octokit_schema: dict[str, Any],
    model_name: str,
    definition_name: str,
    field_name: str,
    required: bool,
    nullable: bool,
    types: set[str],
):
    """Confirmed shared-model drift fields stay aligned with the octokit schema."""

    def type_signature(prop: dict[str, Any]) -> tuple[set[str], bool]:
        """Extract all non-null types/refs from a schema property."""
        if "$ref" in prop:
            return {prop["$ref"].split("/")[-1]}, False

        if "anyOf" in prop or "oneOf" in prop:
            variants = prop.get("anyOf", prop.get("oneOf", []))
            types_seen: set[str] = set()
            is_nullable = False
            for opt in variants:
                if opt.get("type") == "null":
                    is_nullable = True
                elif "$ref" in opt:
                    types_seen.add(opt["$ref"].split("/")[-1])
                elif "type" in opt:
                    types_seen.add(opt["type"])
            return types_seen, is_nullable

        if "allOf" in prop:
            allof_types: set[str] = set()
            for opt in prop["allOf"]:
                if "$ref" in opt:
                    allof_types.add(opt["$ref"].split("/")[-1])
                elif "type" in opt:
                    allof_types.add(opt["type"])
            return allof_types or {"object"}, False

        if "type" in prop:
            t = prop["type"]
            if isinstance(t, list):
                return {x for x in t if x != "null"}, "null" in t
            return {t}, False

        return {"unknown"}, False

    model_class = getattr(gh_module, model_name)
    pydantic_schema = model_class.model_json_schema()
    octokit_def = octokit_schema["definitions"][definition_name]

    pydantic_prop = pydantic_schema["properties"][field_name]
    octokit_prop = octokit_def["properties"][field_name]

    pydantic_types, pydantic_nullable = type_signature(pydantic_prop)
    octokit_types, octokit_nullable = type_signature(octokit_prop)

    assert (field_name in pydantic_schema.get("required", [])) is required
    assert (field_name in octokit_def.get("required", [])) is required
    assert pydantic_nullable is nullable
    assert octokit_nullable is nullable
    assert pydantic_types == types
    assert octokit_types == types


def get_all_pydantic_models() -> list[type[Any]]:
    """Get all Pydantic model classes from the GitHub module."""
    from pydantic import BaseModel

    models: list[type[Any]] = []
    for name in dir(gh_module):
        cls = getattr(gh_module, name)
        if (
            isinstance(cls, type)
            and issubclass(cls, BaseModel)
            and cls.__module__ == gh_module.__name__
        ):
            models.append(cls)
    return models


def test_all_fields_have_descriptions():
    """Verify all Pydantic fields in GitHub SDK have descriptions.

    This ensures documentation is provided for every field, making the SDK
    easier to understand and use.
    """
    models = get_all_pydantic_models()
    missing_descriptions: list[str] = []

    for model in models:
        schema = model.model_json_schema()
        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            # Check if field has a description
            has_description = False

            # Direct description
            if "description" in field_schema:
                has_description = True
            # Description in anyOf/oneOf variants
            elif "anyOf" in field_schema or "oneOf" in field_schema:
                variants = field_schema.get("anyOf", field_schema.get("oneOf", []))
                # Check if any variant or the parent has description
                if any("description" in v for v in variants):
                    has_description = True
            # Description in allOf
            elif "allOf" in field_schema:
                if any("description" in v for v in field_schema["allOf"]):
                    has_description = True

            if not has_description:
                missing_descriptions.append(f"{model.__name__}.{field_name}")

    if missing_descriptions:
        pytest.fail(
            f"Fields missing descriptions ({len(missing_descriptions)}):\n"
            + "\n".join(sorted(missing_descriptions))
        )

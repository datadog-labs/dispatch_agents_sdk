"""Unit tests for Agent UID functionality."""

import pytest

from dispatch_agents.models import Agent, AgentContainerStatus


class TestAgentBuildUid:
    """Tests for Agent.build_uid static method."""

    def test_build_uid_basic(self):
        """Test basic UID construction."""
        uid = Agent.build_uid("org123", "default", "my-agent")
        assert uid == "org123#default#my-agent"

    def test_build_uid_with_special_chars_in_components(self):
        """Test that components with dashes, underscores work."""
        uid = Agent.build_uid("org-123", "dev_namespace", "my-agent-v2")
        assert uid == "org-123#dev_namespace#my-agent-v2"

    def test_build_uid_rejects_hash_in_org_id(self):
        """Test that hash in org_id is rejected."""
        with pytest.raises(ValueError, match="cannot contain '#' separator"):
            Agent.build_uid("org#123", "default", "my-agent")

    def test_build_uid_rejects_hash_in_namespace(self):
        """Test that hash in namespace is rejected."""
        with pytest.raises(ValueError, match="cannot contain '#' separator"):
            Agent.build_uid("org123", "default#prod", "my-agent")

    def test_build_uid_rejects_hash_in_agent_name(self):
        """Test that hash in agent_name is rejected."""
        with pytest.raises(ValueError, match="cannot contain '#' separator"):
            Agent.build_uid("org123", "default", "my#agent")

    def test_build_uid_with_empty_components(self):
        """Test that empty components are allowed (validation happens elsewhere)."""
        # Empty strings are technically valid for build_uid
        # Validation should happen at the model level
        uid = Agent.build_uid("", "", "")
        assert uid == "##"


class TestAgentUidProperty:
    """Tests for Agent.uid computed property."""

    def test_uid_property(self):
        """Test that uid property computes correctly."""
        agent = Agent(
            name="my-agent",
            org_id="org123",
            namespace="default",
            status=AgentContainerStatus.BUILDING,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        assert agent.uid == "org123#default#my-agent"

    def test_uid_property_with_different_org(self):
        """Test uid with different org."""
        agent = Agent(
            name="my-agent",
            org_id="acme-corp",
            namespace="production",
            status=AgentContainerStatus.DEPLOYED,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        assert agent.uid == "acme-corp#production#my-agent"

    def test_uid_uniqueness_across_orgs(self):
        """Test that same agent name in different orgs produces different UIDs."""
        agent1 = Agent(
            name="my-agent",
            org_id="org1",
            namespace="default",
            status=AgentContainerStatus.BUILDING,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        agent2 = Agent(
            name="my-agent",
            org_id="org2",
            namespace="default",
            status=AgentContainerStatus.BUILDING,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        assert agent1.uid != agent2.uid
        assert agent1.uid == "org1#default#my-agent"
        assert agent2.uid == "org2#default#my-agent"

    def test_uid_uniqueness_across_namespaces(self):
        """Test that same agent name in different namespaces produces different UIDs."""
        agent1 = Agent(
            name="my-agent",
            org_id="org123",
            namespace="dev",
            status=AgentContainerStatus.BUILDING,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        agent2 = Agent(
            name="my-agent",
            org_id="org123",
            namespace="prod",
            status=AgentContainerStatus.BUILDING,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        assert agent1.uid != agent2.uid
        assert agent1.uid == "org123#dev#my-agent"
        assert agent2.uid == "org123#prod#my-agent"

    def test_uid_is_computed_not_stored(self):
        """Test that uid is a property and not a stored field."""
        agent = Agent(
            name="my-agent",
            org_id="org123",
            namespace="default",
            status=AgentContainerStatus.BUILDING,
            created_at="2025-01-01T00:00:00Z",
            last_updated="2025-01-01T00:00:00Z",
        )
        # uid should not be in model_dump (not a stored field)
        model_dict = agent.model_dump()
        assert "uid" not in model_dict

        # But should be accessible as a property
        assert agent.uid == "org123#default#my-agent"


class TestAgentCreate:
    """Tests for Agent.create factory method (ensure it still works)."""

    def test_create_sets_uid_correctly(self):
        """Test that Agent.create produces correct uid."""
        agent = Agent.create(
            name="My Test Agent",
            org_id="org123",
            namespace="default",
            status=AgentContainerStatus.BUILDING,
        )
        # Agent name should be sanitized (spaces replaced with dashes, case preserved)
        assert agent.name == "My-Test-Agent"
        # UID should be computed correctly
        assert agent.uid == "org123#default#My-Test-Agent"

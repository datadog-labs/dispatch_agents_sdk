"""Tests for dispatch_agents.__init__ module.

Covers get_data_dir, _init_allowed_prefixes, and _dev_mode_audit_hook.
"""

import os
from pathlib import Path

import pytest


class TestGetDataDir:
    def test_production(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_DEV_DATA_DIR", raising=False)
        from dispatch_agents import get_data_dir

        assert get_data_dir() == Path("/data")

    def test_dev_mode(self, monkeypatch, tmp_path):
        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        from dispatch_agents import get_data_dir

        assert get_data_dir() == dev_dir / "data"


class TestInitAllowedPrefixes:
    def test_includes_data_dir(self, monkeypatch, tmp_path):
        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        from dispatch_agents import _init_allowed_prefixes

        prefixes = _init_allowed_prefixes()
        # Data dir should be present (resolved form of dev_dir/data)
        data_dir = str((dev_dir / "data").resolve())
        assert any(
            data_dir.startswith(p) or p.startswith(data_dir) or data_dir == p
            for p in prefixes
        )

    def test_includes_tmp(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_DEV_DATA_DIR", raising=False)

        from dispatch_agents import _init_allowed_prefixes

        prefixes = _init_allowed_prefixes()
        resolved_tmp = str(Path("/tmp").resolve())
        assert any(p == resolved_tmp or resolved_tmp.startswith(p) for p in prefixes)

    def test_includes_agent_folder(self, monkeypatch, tmp_path):
        agent_folder = tmp_path / "my-agent"
        dev_dir = agent_folder / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        from dispatch_agents import _init_allowed_prefixes

        prefixes = _init_allowed_prefixes()
        resolved_agent = str(agent_folder.resolve())
        assert any(
            resolved_agent.startswith(p) or p == resolved_agent for p in prefixes
        )

    def test_no_dev_dir(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_DEV_DATA_DIR", raising=False)

        from dispatch_agents import _init_allowed_prefixes

        prefixes = _init_allowed_prefixes()
        # Should still have /data and /tmp
        assert len(prefixes) >= 2


class TestDevModeAuditHook:
    def test_ignores_non_open_events(self):
        from dispatch_agents import _dev_mode_audit_hook

        # Should not raise for non-open events
        _dev_mode_audit_hook("import", ("os",))

    def test_ignores_read_mode(self, monkeypatch, tmp_path):
        from dispatch_agents import _dev_mode_audit_hook

        _dev_mode_audit_hook("open", ("/etc/hosts", "r"))

    def test_ignores_empty_args(self):
        from dispatch_agents import _dev_mode_audit_hook

        _dev_mode_audit_hook("open", ())

    def test_ignores_short_args(self):
        from dispatch_agents import _dev_mode_audit_hook

        _dev_mode_audit_hook("open", ("/some/path",))

    def test_blocks_write_outside_allowed(self, monkeypatch, tmp_path):
        import dispatch_agents

        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        # Set up allowed prefixes
        old_prefixes = dispatch_agents._audit_hook_allowed_prefixes
        old_blocked = dispatch_agents._audit_hook_blocked.copy()
        try:
            dispatch_agents._audit_hook_allowed_prefixes = (
                dispatch_agents._init_allowed_prefixes()
            )
            dispatch_agents._audit_hook_blocked.clear()

            with pytest.raises(dispatch_agents.DisallowedWriteError):
                dispatch_agents._dev_mode_audit_hook("open", ("/etc/bad_file", "w"))
        finally:
            dispatch_agents._audit_hook_allowed_prefixes = old_prefixes
            dispatch_agents._audit_hook_blocked = old_blocked

    def test_allows_data_dir_write(self, monkeypatch, tmp_path):
        import dispatch_agents

        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        old_prefixes = dispatch_agents._audit_hook_allowed_prefixes
        try:
            dispatch_agents._audit_hook_allowed_prefixes = (
                dispatch_agents._init_allowed_prefixes()
            )

            data_path = str((dev_dir / "data" / "test.txt").resolve())
            # Should not raise
            dispatch_agents._dev_mode_audit_hook("open", (data_path, "w"))
        finally:
            dispatch_agents._audit_hook_allowed_prefixes = old_prefixes

    def test_allows_tmp_write(self, monkeypatch, tmp_path):
        import dispatch_agents

        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        old_prefixes = dispatch_agents._audit_hook_allowed_prefixes
        try:
            dispatch_agents._audit_hook_allowed_prefixes = (
                dispatch_agents._init_allowed_prefixes()
            )

            tmp_file = str(Path("/tmp/test_file.txt").resolve())
            dispatch_agents._dev_mode_audit_hook("open", (tmp_file, "w"))
        finally:
            dispatch_agents._audit_hook_allowed_prefixes = old_prefixes

    def test_blocks_integer_write_flags(self, monkeypatch, tmp_path):
        import dispatch_agents

        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        old_prefixes = dispatch_agents._audit_hook_allowed_prefixes
        old_blocked = dispatch_agents._audit_hook_blocked.copy()
        try:
            dispatch_agents._audit_hook_allowed_prefixes = (
                dispatch_agents._init_allowed_prefixes()
            )
            dispatch_agents._audit_hook_blocked.clear()

            with pytest.raises(dispatch_agents.DisallowedWriteError):
                dispatch_agents._dev_mode_audit_hook(
                    "open", ("/etc/secret", os.O_WRONLY)
                )
        finally:
            dispatch_agents._audit_hook_allowed_prefixes = old_prefixes
            dispatch_agents._audit_hook_blocked = old_blocked

    def test_repeated_block_shorter_message(self, monkeypatch, tmp_path):
        import dispatch_agents

        dev_dir = tmp_path / ".dispatch" / "dev-data"
        dev_dir.mkdir(parents=True)
        monkeypatch.setenv("DISPATCH_DEV_DATA_DIR", str(dev_dir))

        old_prefixes = dispatch_agents._audit_hook_allowed_prefixes
        old_blocked = dispatch_agents._audit_hook_blocked.copy()
        try:
            dispatch_agents._audit_hook_allowed_prefixes = (
                dispatch_agents._init_allowed_prefixes()
            )
            dispatch_agents._audit_hook_blocked.clear()

            # First attempt - detailed message
            with pytest.raises(
                dispatch_agents.DisallowedWriteError, match="outside allowed"
            ):
                dispatch_agents._dev_mode_audit_hook("open", ("/etc/repeated", "w"))

            # Second attempt - shorter message
            with pytest.raises(
                dispatch_agents.DisallowedWriteError, match="repeated attempt"
            ):
                dispatch_agents._dev_mode_audit_hook("open", ("/etc/repeated", "w"))
        finally:
            dispatch_agents._audit_hook_allowed_prefixes = old_prefixes
            dispatch_agents._audit_hook_blocked = old_blocked

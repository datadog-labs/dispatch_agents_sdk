"""Tests for dispatch_agents._internal.logging_config module.

Log level resolves as: ``DISPATCH_LOG_LEVEL`` env override → the ``log_level``
field in dispatch.yaml (read through the runtime config model) → WARNING. Tests
point ``DISPATCH_CONFIG_PATH`` at a temp config file and clear the
runtime-config cache between cases.
"""

import logging

import pytest


def _write_config(tmp_path, body: str | None) -> str:
    """Write a dispatch.yaml with the given body and return its path."""
    path = tmp_path / "dispatch.yaml"
    if body is not None:
        path.write_text(body, encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def _clear_runtime_config_cache():
    """Ensure each test reads a fresh runtime config."""
    from dispatch_agents.config import _load_runtime_config

    _load_runtime_config.cache_clear()
    yield
    _load_runtime_config.cache_clear()


def _point_at(monkeypatch, path: str) -> None:
    from dispatch_agents.config import _load_runtime_config

    # Clear the operational override so config-driven cases are deterministic
    # regardless of the ambient environment.
    monkeypatch.delenv("DISPATCH_LOG_LEVEL", raising=False)
    monkeypatch.setenv("DISPATCH_CONFIG_PATH", path)
    _load_runtime_config.cache_clear()


class TestGetLogLevel:
    def test_default_is_warning_when_no_file(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, str(tmp_path / "missing.yaml"))

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.WARNING

    def test_default_is_warning_when_key_absent(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "namespace: dev\n"))

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.WARNING

    def test_explicit_debug(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: DEBUG\n"))

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.DEBUG

    def test_explicit_info(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: INFO\n"))

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.INFO

    def test_explicit_error(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: ERROR\n"))

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.ERROR

    def test_case_insensitive(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: debug\n"))

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.DEBUG


class TestEnvOverride:
    """DISPATCH_LOG_LEVEL is an operational override that wins over config."""

    def test_env_override_without_config_file(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, str(tmp_path / "missing.yaml"))
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "DEBUG")

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.DEBUG

    def test_env_override_beats_config(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: ERROR\n"))
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "DEBUG")

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.DEBUG

    def test_env_override_case_insensitive(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, str(tmp_path / "missing.yaml"))
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "info")

        from dispatch_agents._internal.logging_config import _get_log_level

        assert _get_log_level() == logging.INFO


class TestConfigureLogging:
    def test_sets_level(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: INFO\n"))

        import dispatch_agents._internal.logging_config as lc

        lc._logging_configured = False
        lc.configure_logging(force=True)

        sdk_logger = logging.getLogger(lc.SDK_LOGGER_NAME)
        assert sdk_logger.level == logging.INFO

    def test_idempotent_without_force(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, None))

        import dispatch_agents._internal.logging_config as lc

        lc._logging_configured = False

        sdk_logger = logging.getLogger(lc.SDK_LOGGER_NAME)
        sdk_logger.handlers.clear()

        lc.configure_logging(force=True)
        handler_count = len(sdk_logger.handlers)

        # Second call without force should not add handlers
        lc.configure_logging()
        assert len(sdk_logger.handlers) == handler_count


class TestGetLogger:
    def test_child_logger(self):
        from dispatch_agents._internal.logging_config import SDK_LOGGER_NAME, get_logger

        logger = get_logger("grpc_server")
        assert logger.name == f"{SDK_LOGGER_NAME}.grpc_server"

    def test_root_logger(self):
        from dispatch_agents._internal.logging_config import SDK_LOGGER_NAME, get_logger

        logger = get_logger()
        assert logger.name == SDK_LOGGER_NAME

    def test_fully_qualified_name_passthrough(self):
        from dispatch_agents._internal.logging_config import SDK_LOGGER_NAME, get_logger

        fqn = f"{SDK_LOGGER_NAME}.some_module"
        logger = get_logger(fqn)
        assert logger.name == fqn


class TestIsVerbose:
    def test_false_by_default(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, None))

        from dispatch_agents._internal.logging_config import is_verbose

        assert is_verbose() is False

    def test_true_when_debug(self, monkeypatch, tmp_path):
        _point_at(monkeypatch, _write_config(tmp_path, "log_level: DEBUG\n"))

        from dispatch_agents._internal.logging_config import is_verbose

        assert is_verbose() is True

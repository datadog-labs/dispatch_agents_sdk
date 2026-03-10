"""Tests for dispatch_agents.logging_config module."""

import logging


class TestParseBoolEnv:
    def test_truthy_values(self):
        from dispatch_agents.logging_config import _parse_bool_env

        for val in ("1", "true", "yes", "on", "True", "YES", "ON"):
            assert _parse_bool_env(val) is True, f"Expected True for {val!r}"

    def test_falsy_values(self):
        from dispatch_agents.logging_config import _parse_bool_env

        assert _parse_bool_env(None) is False
        assert _parse_bool_env("0") is False
        assert _parse_bool_env("no") is False
        assert _parse_bool_env("false") is False
        assert _parse_bool_env("off") is False
        assert _parse_bool_env("") is False


class TestGetLogLevel:
    def test_default_is_warning(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOG_LEVEL", raising=False)
        monkeypatch.delenv("DISPATCH_VERBOSE", raising=False)

        from dispatch_agents.logging_config import _get_log_level

        assert _get_log_level() == logging.WARNING

    def test_explicit_debug(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "DEBUG")
        monkeypatch.delenv("DISPATCH_VERBOSE", raising=False)

        from dispatch_agents.logging_config import _get_log_level

        assert _get_log_level() == logging.DEBUG

    def test_explicit_info(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "INFO")

        from dispatch_agents.logging_config import _get_log_level

        assert _get_log_level() == logging.INFO

    def test_explicit_error(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "ERROR")

        from dispatch_agents.logging_config import _get_log_level

        assert _get_log_level() == logging.ERROR

    def test_verbose_flag(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOG_LEVEL", raising=False)
        monkeypatch.setenv("DISPATCH_VERBOSE", "1")

        from dispatch_agents.logging_config import _get_log_level

        assert _get_log_level() == logging.DEBUG

    def test_explicit_level_takes_priority_over_verbose(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "ERROR")
        monkeypatch.setenv("DISPATCH_VERBOSE", "1")

        from dispatch_agents.logging_config import _get_log_level

        assert _get_log_level() == logging.ERROR


class TestConfigureLogging:
    def test_sets_level(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "INFO")
        monkeypatch.delenv("DISPATCH_VERBOSE", raising=False)

        import dispatch_agents.logging_config as lc

        lc._logging_configured = False
        lc.configure_logging(force=True)

        sdk_logger = logging.getLogger(lc.SDK_LOGGER_NAME)
        assert sdk_logger.level == logging.INFO

    def test_idempotent_without_force(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOG_LEVEL", raising=False)
        monkeypatch.delenv("DISPATCH_VERBOSE", raising=False)

        import dispatch_agents.logging_config as lc

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
        from dispatch_agents.logging_config import SDK_LOGGER_NAME, get_logger

        logger = get_logger("grpc_server")
        assert logger.name == f"{SDK_LOGGER_NAME}.grpc_server"

    def test_root_logger(self):
        from dispatch_agents.logging_config import SDK_LOGGER_NAME, get_logger

        logger = get_logger()
        assert logger.name == SDK_LOGGER_NAME

    def test_fully_qualified_name_passthrough(self):
        from dispatch_agents.logging_config import SDK_LOGGER_NAME, get_logger

        fqn = f"{SDK_LOGGER_NAME}.some_module"
        logger = get_logger(fqn)
        assert logger.name == fqn


class TestIsVerbose:
    def test_false_by_default(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOG_LEVEL", raising=False)
        monkeypatch.delenv("DISPATCH_VERBOSE", raising=False)

        from dispatch_agents.logging_config import is_verbose

        assert is_verbose() is False

    def test_true_when_debug(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_LOG_LEVEL", "DEBUG")

        from dispatch_agents.logging_config import is_verbose

        assert is_verbose() is True

    def test_true_when_verbose(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOG_LEVEL", raising=False)
        monkeypatch.setenv("DISPATCH_VERBOSE", "1")

        from dispatch_agents.logging_config import is_verbose

        assert is_verbose() is True

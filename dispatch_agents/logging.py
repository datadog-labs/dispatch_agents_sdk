"""Logging configuration for the Dispatch Agents SDK.

The SDK log verbosity (DEBUG/INFO/WARNING/ERROR, case-insensitive) is resolved
in priority order:

1. ``DISPATCH_LOG_LEVEL`` environment variable — an operational override set by
   the platform/CLI (e.g. ``dispatch agent run --verbose``). This is a logging
   knob, not a config-file channel, so it does not read ``dispatch.yaml``.
2. The ``log_level`` field in ``dispatch.yaml``, read through the runtime config
   model (:mod:`dispatch_agents.config`), which is the only reader of the file.
3. WARNING by default, hiding routine info/debug messages.

Logging sits above config in the dependency graph: config loading never logs,
so logging can depend on config without a cycle.
"""

import logging
import os
import sys

__all__ = [
    "SDK_LOGGER_NAME",
    "configure_logging",
    "get_logger",
    "is_verbose",
]

# SDK logger namespace
SDK_LOGGER_NAME = "dispatch_agents"

# Check if we've already configured logging
_logging_configured = False


def _get_log_level() -> int:
    """Resolve the log level from the env override, then config, then default.

    Maps the resolved level name to a :mod:`logging` level. Defaults to WARNING
    when unset, unknown, or unavailable.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    # Operational override (platform/CLI-set), then the dispatch.yaml field.
    level_str = os.environ.get("DISPATCH_LOG_LEVEL")
    if not level_str:
        # Lazy import: many SDK modules call get_logger() at import time, so a
        # module-level import of config would risk a cycle. Config loading is
        # logger-free, so importing it here cannot recurse back into logging.
        try:
            from dispatch_agents.config import config as _config

            level_str = _config.log_level
        except Exception:
            level_str = None

    if level_str and (mapped := level_map.get(level_str.upper())) is not None:
        return mapped

    # Default: WARNING to minimize noise
    # Users see errors and warnings, but not routine info/debug
    return logging.WARNING


def configure_logging(force: bool = False) -> None:
    """Configure logging for the Dispatch Agents SDK.

    This sets up the SDK logger with appropriate level and format.
    Called automatically when the SDK is imported, but can be called
    again with force=True to reconfigure.

    Args:
        force: If True, reconfigure even if already configured
    """
    global _logging_configured

    if _logging_configured and not force:
        return

    # Get the SDK root logger
    sdk_logger = logging.getLogger(SDK_LOGGER_NAME)

    # Set the log level
    level = _get_log_level()
    sdk_logger.setLevel(level)

    # Only add handler if none exist (avoid duplicate handlers on force)
    if not sdk_logger.handlers:
        # Create a handler that writes to stderr
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        # Format: simple for normal use, more detail for debug
        if level == logging.DEBUG:
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            formatter = logging.Formatter("%(levelname)s: %(message)s")

        handler.setFormatter(formatter)
        sdk_logger.addHandler(handler)

    # Prevent propagation to root logger (avoid duplicate messages)
    sdk_logger.propagate = False

    _logging_configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for SDK modules.

    Args:
        name: Optional module name (e.g., __name__). If provided,
              creates a child logger under SDK_LOGGER_NAME.

    Returns:
        Configured logger instance
    """
    # Ensure logging is configured
    configure_logging()

    if name:
        # Create child logger: dispatch_agents.grpc_server, etc.
        if name.startswith(SDK_LOGGER_NAME):
            return logging.getLogger(name)
        return logging.getLogger(f"{SDK_LOGGER_NAME}.{name}")
    return logging.getLogger(SDK_LOGGER_NAME)


def is_verbose() -> bool:
    """Check if verbose logging is enabled.

    Returns:
        True if the configured log level is DEBUG
    """
    return _get_log_level() == logging.DEBUG

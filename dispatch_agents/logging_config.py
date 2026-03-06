"""Logging configuration for the Dispatch Agents SDK.

Controls the verbosity of SDK logging output via environment variables:
- DISPATCH_VERBOSE=1 or DISPATCH_VERBOSE=true: Enable verbose/debug logging
- DISPATCH_LOG_LEVEL=DEBUG/INFO/WARNING/ERROR: Set specific log level

By default, the SDK logs at INFO level, hiding debug messages like
re-subscription events. Enable verbose mode for debugging.
"""

import logging
import os
import sys

# SDK logger namespace
SDK_LOGGER_NAME = "dispatch_agents"

# Check if we've already configured logging
_logging_configured = False


def _parse_bool_env(value: str | None) -> bool:
    """Parse boolean from environment variable value."""
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "on")


def _get_log_level() -> int:
    """Determine the log level from environment variables.

    Priority:
    1. DISPATCH_LOG_LEVEL (explicit level)
    2. DISPATCH_VERBOSE (boolean toggle for DEBUG)
    3. Default: WARNING (minimal output)
    """
    # Check explicit log level first
    log_level_str = os.environ.get("DISPATCH_LOG_LEVEL", "").upper()
    if log_level_str:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if log_level_str in level_map:
            return level_map[log_level_str]

    # Check verbose flag
    if _parse_bool_env(os.environ.get("DISPATCH_VERBOSE")):
        return logging.DEBUG

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
        True if DISPATCH_VERBOSE is set or log level is DEBUG
    """
    return _get_log_level() == logging.DEBUG

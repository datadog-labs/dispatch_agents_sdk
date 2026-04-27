import os
import sys
import tempfile
from pathlib import Path

# Add SDK root to Python path so generated protobuf files can import
# using their package structure (e.g., "from agentservice.v1 import ...")
_sdk_root = str(Path(__file__).parent.parent)
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)


def get_data_dir() -> Path:
    """Get the data directory for persistent storage.

    In production, this returns /data (EFS mount point).
    In local dev mode (`dispatch agent dev`), this returns the mock data directory
    set by DISPATCH_DEV_DATA_DIR environment variable.

    Returns:
        Path to the data directory

    Example:
        from dispatch_agents import get_data_dir

        DATA_DIR = get_data_dir()
        my_file = DATA_DIR / "myfile.txt"
        my_file.write_text("hello")
    """
    dev_data_dir = os.environ.get("DISPATCH_DEV_DATA_DIR")
    if dev_data_dir:
        # In dev mode, use the mock data directory
        return Path(dev_data_dir) / "data"
    # In production, use the EFS mount at /data
    return Path("/data")


# =============================================================================
# Dev Mode Audit Hook - Restricts writes to allowed directories
# =============================================================================


class DisallowedWriteError(Exception):
    """Raised when an agent attempts to write outside allowed directories in dev mode."""

    pass


_audit_hook_blocked: set[str] = set()  # Track blocked paths to avoid duplicate errors
_audit_hook_allowed_prefixes: list[str] = []  # Computed once at startup


def _init_allowed_prefixes() -> list[str]:
    """Compute the list of allowed write prefixes for dev mode.

    Allowed locations:
    1. get_data_dir() - the mock data directory
    2. /tmp - temporary files (and system temp directory)
    3. Agent folder - where the agent code lives (derived from DISPATCH_DEV_DATA_DIR)

    All paths are resolved to their canonical form to handle symlinks
    (e.g., macOS /var -> /private/var).
    """
    allowed = []

    def resolve_path(p: str) -> str:
        """Resolve path to canonical form, handling symlinks."""
        try:
            return str(Path(p).resolve())
        except (OSError, ValueError):
            return p

    # 1. The data directory from get_data_dir()
    allowed.append(resolve_path(str(get_data_dir())))

    # 2. Temp directories - include both standard paths and the system's actual temp
    # Resolve all paths to handle macOS symlinks (/var -> /private/var)
    allowed.append(resolve_path("/tmp"))
    allowed.append(resolve_path("/var/tmp"))
    # macOS temp dirs (may be redundant after resolution, but included for safety)
    allowed.append(resolve_path("/private/tmp"))
    allowed.append(resolve_path("/private/var/tmp"))
    # System temp directory (handles macOS /var/folders/... and other platforms)
    allowed.append(resolve_path(tempfile.gettempdir()))

    # 3. Agent folder - DISPATCH_DEV_DATA_DIR is {agent_path}/.dispatch/dev-data
    dev_data_dir = os.environ.get("DISPATCH_DEV_DATA_DIR", "")
    if dev_data_dir:
        # Go up two levels: dev-data -> .dispatch -> agent_folder
        agent_folder = str(Path(dev_data_dir).parent.parent)
        allowed.append(resolve_path(agent_folder))

    return allowed


def _dev_mode_audit_hook(event: str, args: tuple) -> None:
    """Block file operations that target paths outside allowed directories.

    In dev mode, agents should only write to:
    - get_data_dir() (mock persistent storage)
    - /tmp (temporary files)
    - The agent's own folder

    This protects the developer's machine from accidental writes to arbitrary locations.
    Use --allow-arbitrary-writes flag or DISPATCH_ALLOW_ARBITRARY_WRITES=1 to disable.
    """
    # Only care about file open operations that could write
    if event != "open":
        return

    if not args or len(args) < 2:
        return

    raw_path = str(args[0])
    mode = args[1] if len(args) > 1 else ""

    # Only check write operations (mode contains 'w', 'a', 'x', or '+')
    # Skip read-only operations
    if isinstance(mode, str):
        if "w" not in mode and "a" not in mode and "x" not in mode and "+" not in mode:
            return
    elif isinstance(mode, int):
        # For integer flags, check if it's a write mode
        # os.O_WRONLY = 1, os.O_RDWR = 2, os.O_APPEND = 1024, os.O_CREAT = 512
        import os as os_module

        write_flags = getattr(os_module, "O_WRONLY", 1) | getattr(
            os_module, "O_RDWR", 2
        )
        if not (mode & write_flags):
            return

    # Resolve relative paths to absolute (e.g., "../test" -> "/absolute/path/test")
    # This ensures path traversal attacks like "../../../etc/passwd" are caught
    try:
        path = str(Path(raw_path).resolve())
    except (OSError, ValueError):
        # If path resolution fails, use the raw path
        path = raw_path

    # Check if path is within any allowed prefix
    for prefix in _audit_hook_allowed_prefixes:
        if path.startswith(prefix):
            return  # Allowed

    # Track whether we've already shown the detailed error for this path
    # (to reduce noise), but ALWAYS raise the error to block the write
    show_details = path not in _audit_hook_blocked
    _audit_hook_blocked.add(path)

    if show_details:
        allowed_locations = ", ".join(_audit_hook_allowed_prefixes[:3])  # Show first 3
        raise DisallowedWriteError(
            f"Write operation to '{path}' blocked - outside allowed directories.\n"
            f"In dev mode, writes are only allowed to: {allowed_locations}\n"
            "Use get_data_dir() for persistent storage.\n"
            "To disable this check, run with: dispatch agent dev --allow-arbitrary-writes"
        )
    else:
        # Subsequent attempts to the same path: still block, but shorter message
        raise DisallowedWriteError(f"Write to '{path}' blocked (repeated attempt)")


from .agent_service import AgentServiceClient
from .config import _runtime as config
from .events import (
    HANDLER_METADATA,
    REGISTERED_HANDLERS,
    TOPIC_HANDLERS,
    AsyncHandler,
    BasePayload,
    HandlerMetadata,
    dispatch_message,
    emit_event,
    fn,
    get_current_invocation_id,
    get_current_parent_id,
    get_current_trace_id,
    get_handler_metadata,
    get_handler_schemas,
    get_invocation_id_for_trace,
    init,
    invoke,
    on,
    run_init_hook,
)
from .invocation import InvocationStatus
from .llm import LLMFunctionCall, LLMToolCall, extra_headers, get_extra_llm_headers
from .mcp import McpHttpServerConfig, get_mcp_client, get_mcp_servers_config
from .memory import MemoryClient, memory
from .models import (
    Agent,
    AgentContainerStatus,
    BaseMessage,
    FeedbackSentiment,
    FeedbackType,
    FunctionMessage,
    JsonSchema,
    KVGetResponse,
    KVListResponse,
    KVMemoryRecord,
    LLMCallMessage,
    MemoryWriteResponse,
    Message,
    ScheduleMessage,
    SessionGetResponse,
    StrictBaseModel,
    TopicMessage,
)

__all__ = [
    # runtime config
    "config",
    # storage and dev mode isolation
    "get_data_dir",
    "DisallowedWriteError",
    # agent service
    "AgentServiceClient",
    # events - decorators and client functions
    "on",
    "init",
    "fn",
    "invoke",
    "dispatch_message",
    "emit_event",
    "run_init_hook",
    # events - context and metadata
    "get_current_trace_id",
    "get_current_invocation_id",
    "get_current_parent_id",
    "get_invocation_id_for_trace",
    "get_handler_schemas",
    "get_handler_metadata",
    # events - registries and types
    "REGISTERED_HANDLERS",  # handler_name -> AsyncHandler
    "HANDLER_METADATA",  # handler_name -> HandlerMetadata
    "TOPIC_HANDLERS",  # topic -> list of handler_names
    "AsyncHandler",  # Type alias for handler functions
    "HandlerMetadata",  # Pydantic model for handler metadata
    "BasePayload",
    # invocation
    "InvocationStatus",
    # models - Message types (discriminated union pattern)
    "BaseMessage",
    "TopicMessage",
    "FunctionMessage",
    "ScheduleMessage",
    "LLMCallMessage",
    "Message",  # Union type alias (TopicMessage | FunctionMessage | ScheduleMessage)
    "Agent",
    "AgentContainerStatus",
    "StrictBaseModel",
    # memory
    "MemoryClient",
    "memory",
    "MemoryWriteResponse",
    "KVGetResponse",
    "KVListResponse",
    "KVMemoryRecord",
    "SessionGetResponse",
    # type aliases
    "JsonSchema",
    # feedback types
    "FeedbackType",
    "FeedbackSentiment",
    # mcp
    "get_mcp_client",
    "get_mcp_servers_config",
    "McpHttpServerConfig",
    # llm
    "LLMFunctionCall",
    "LLMToolCall",
    "extra_headers",
    "get_extra_llm_headers",
]

# =============================================================================
# Install Dev Mode Audit Hook (MUST be after all imports to avoid blocking .pyc writes)
# =============================================================================

# Install the audit hook only in dev mode, unless arbitrary writes are explicitly allowed
if os.environ.get("DISPATCH_DEV_DATA_DIR") and not os.environ.get(
    "DISPATCH_ALLOW_ARBITRARY_WRITES"
):
    # IMPORTANT: Initialize allowed prefixes BEFORE installing the hook.
    # This avoids recursive calls when tempfile.gettempdir() triggers file operations.
    _audit_hook_allowed_prefixes = _init_allowed_prefixes()
    sys.addaudithook(_dev_mode_audit_hook)

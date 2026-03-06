"""Storage tester agent - verifies persist_storage behavior.

This agent provides functions to:
1. Write unique files to /data and working directory on each invocation
2. List directory contents
3. Touch files at arbitrary paths (for access testing)
4. Scan directories recursively (for access boundary testing)
"""

import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from dispatch_agents import BasePayload, fn


class WriteAndListRequest(BasePayload):
    """Input for write_and_list - no parameters needed."""

    pass


class WriteAndListResponse(BasePayload):
    """Output showing what was written and directory contents."""

    unique_id: str
    timestamp: str
    data_dir_exists: bool
    data_dir_writable: bool
    data_dir_contents: list[str]
    cwd: str
    cwd_contents: list[str]
    files_written: list[str]
    errors: list[str]


@fn()
async def write_and_list(payload: WriteAndListRequest) -> WriteAndListResponse:
    """Write unique files to /data and cwd, then list both directories.

    This is the main test function - invoke it before/after redeploy to verify
    whether files persist.
    """
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    errors: list[str] = []
    files_written: list[str] = []

    data_dir = Path("/data")
    cwd = Path.cwd()

    # Check /data status
    data_dir_exists = data_dir.exists()
    data_dir_writable = data_dir_exists and os.access(str(data_dir), os.W_OK)

    # Try to write to /data
    if data_dir_writable:
        try:
            data_file = data_dir / f"test-{unique_id}.txt"
            data_file.write_text(f"Written at {timestamp}\nID: {unique_id}\n")
            files_written.append(str(data_file))
        except Exception as e:
            errors.append(f"Failed to write to /data: {e}")
    elif data_dir_exists:
        errors.append("/data exists but is not writable")
    else:
        errors.append("/data directory does not exist")

    # Try to write to cwd
    try:
        cwd_file = cwd / f"cwd-test-{unique_id}.txt"
        cwd_file.write_text(f"Written at {timestamp}\nID: {unique_id}\n")
        files_written.append(str(cwd_file))
    except Exception as e:
        errors.append(f"Failed to write to cwd: {e}")

    # List /data contents
    data_dir_contents: list[str] = []
    if data_dir_exists:
        try:
            data_dir_contents = sorted([f.name for f in data_dir.iterdir()])
        except Exception as e:
            errors.append(f"Failed to list /data: {e}")

    # List cwd contents
    cwd_contents: list[str] = []
    try:
        cwd_contents = sorted([f.name for f in cwd.iterdir()])
    except Exception as e:
        errors.append(f"Failed to list cwd: {e}")

    return WriteAndListResponse(
        unique_id=unique_id,
        timestamp=timestamp,
        data_dir_exists=data_dir_exists,
        data_dir_writable=data_dir_writable,
        data_dir_contents=data_dir_contents,
        cwd=str(cwd),
        cwd_contents=cwd_contents,
        files_written=files_written,
        errors=errors,
    )


class ListDirRequest(BasePayload):
    """Input for list_dir function."""

    path: str


class ListDirResponse(BasePayload):
    """Output showing directory listing."""

    path: str
    exists: bool
    is_dir: bool
    readable: bool
    contents: list[str]
    error: str | None


@fn()
async def list_dir(payload: ListDirRequest) -> ListDirResponse:
    """List contents of any directory path.

    Use this to explore what the agent can see in the filesystem.
    """
    path = Path(payload.path)

    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    readable = os.access(str(path), os.R_OK) if exists else False
    contents: list[str] = []
    error: str | None = None

    if exists and is_dir and readable:
        try:
            contents = sorted([f.name for f in path.iterdir()])
        except Exception as e:
            error = str(e)
    elif not exists:
        error = "Path does not exist"
    elif not is_dir:
        error = "Path is not a directory"
    elif not readable:
        error = "Path is not readable"

    return ListDirResponse(
        path=str(path),
        exists=exists,
        is_dir=is_dir,
        readable=readable,
        contents=contents,
        error=error,
    )


class TouchFileRequest(BasePayload):
    """Input for touch_file function."""

    path: str


class TouchFileResponse(BasePayload):
    """Output showing touch result."""

    path: str
    success: bool
    error: str | None


@fn()
async def touch_file(payload: TouchFileRequest) -> TouchFileResponse:
    """Touch (create) a file at the given path.

    Use this to test write access to various directories.
    """
    path = Path(payload.path)
    error: str | None = None
    success = False

    try:
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        success = True
    except Exception as e:
        error = str(e)

    return TouchFileResponse(
        path=str(path),
        success=success,
        error=error,
    )


class ScanDirRequest(BasePayload):
    """Input for scan_dir function."""

    path: str
    max_depth: int = 3
    max_files: int = 100


class ScanDirResponse(BasePayload):
    """Output showing recursive directory scan."""

    path: str
    files_found: list[str]
    dirs_found: list[str]
    total_files: int
    total_dirs: int
    truncated: bool
    errors: list[str]


@fn()
async def scan_dir(payload: ScanDirRequest) -> ScanDirResponse:
    """Recursively scan a directory up to max_depth levels.

    Use this to explore the full filesystem structure and test access boundaries.
    """
    root = Path(payload.path)
    files_found: list[str] = []
    dirs_found: list[str] = []
    errors: list[str] = []
    truncated = False

    def scan(path: Path, depth: int) -> None:
        nonlocal truncated

        if depth > payload.max_depth:
            return

        if len(files_found) + len(dirs_found) >= payload.max_files:
            truncated = True
            return

        try:
            for item in path.iterdir():
                if truncated:
                    return

                rel_path = str(item.relative_to(root))

                if item.is_dir():
                    dirs_found.append(rel_path)
                    scan(item, depth + 1)
                else:
                    files_found.append(rel_path)

                if len(files_found) + len(dirs_found) >= payload.max_files:
                    truncated = True
                    return
        except PermissionError:
            errors.append(f"Permission denied: {path}")
        except Exception as e:
            errors.append(f"Error scanning {path}: {e}")

    if root.exists() and root.is_dir():
        scan(root, 0)
    else:
        errors.append(f"Path does not exist or is not a directory: {root}")

    return ScanDirResponse(
        path=str(root),
        files_found=sorted(files_found),
        dirs_found=sorted(dirs_found),
        total_files=len(files_found),
        total_dirs=len(dirs_found),
        truncated=truncated,
        errors=errors,
    )


class RunLsRequest(BasePayload):
    """Input for run_ls function."""

    path: str
    flags: str = "-la"


class RunLsResponse(BasePayload):
    """Output showing ls command result."""

    path: str
    output: str
    return_code: int
    error: str | None


@fn()
async def run_ls(payload: RunLsRequest) -> RunLsResponse:
    """Run the actual ls command on a path.

    This gives you the raw ls output including permissions, sizes, etc.
    """
    error: str | None = None
    output = ""
    return_code = -1

    try:
        result = subprocess.run(
            ["ls", payload.flags, payload.path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        error = "Command timed out"
    except Exception as e:
        error = str(e)

    return RunLsResponse(
        path=payload.path,
        output=output,
        return_code=return_code,
        error=error,
    )


class GetMountInfoRequest(BasePayload):
    """Input for get_mount_info - no parameters needed."""

    pass


class GetMountInfoResponse(BasePayload):
    """Output showing mount information."""

    mounts: str
    df_output: str
    errors: list[str]


@fn()
async def get_mount_info(payload: GetMountInfoRequest) -> GetMountInfoResponse:
    """Get filesystem mount information.

    Shows what filesystems are mounted, useful for verifying EFS mount.
    """
    errors: list[str] = []
    mounts = ""
    df_output = ""

    # Read /proc/mounts
    try:
        mounts = Path("/proc/mounts").read_text()
    except Exception as e:
        errors.append(f"Failed to read /proc/mounts: {e}")

    # Run df -h
    try:
        result = subprocess.run(
            ["df", "-h"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        df_output = result.stdout + result.stderr
    except Exception as e:
        errors.append(f"Failed to run df: {e}")

    return GetMountInfoResponse(
        mounts=mounts,
        df_output=df_output,
        errors=errors,
    )


class RunCommandRequest(BasePayload):
    """Input for run_command function."""

    command: str
    timeout: int = 30


class RunCommandResponse(BasePayload):
    """Output showing command result."""

    command: str
    stdout: str
    stderr: str
    return_code: int
    error: str | None


@fn()
async def run_command(payload: RunCommandRequest) -> RunCommandResponse:
    """Run an arbitrary shell command.

    Use this for ad-hoc exploration and testing.
    """
    error: str | None = None
    stdout = ""
    stderr = ""
    return_code = -1

    try:
        result = subprocess.run(
            payload.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=payload.timeout,
        )
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        error = f"Command timed out after {payload.timeout}s"
    except Exception as e:
        error = str(e)

    return RunCommandResponse(
        command=payload.command,
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        error=error,
    )

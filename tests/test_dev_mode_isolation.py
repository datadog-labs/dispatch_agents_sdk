"""Tests for dev mode filesystem isolation.

These tests verify that agents running in dev mode cannot write
to arbitrary locations on the developer's machine.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

# ============================================================================
# Test: get_data_dir() function
# ============================================================================


class TestGetDataDir:
    """Tests for get_data_dir() function."""

    def test_production_mode_returns_data(self):
        """In production mode (no DISPATCH_DEV_DATA_DIR), returns /data."""
        # Import fresh without the env var
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("DISPATCH_DEV_DATA_DIR", None)

            # Re-import to get fresh function behavior
            from dispatch_agents import get_data_dir

            result = get_data_dir()
            assert result == Path("/data")

    def test_dev_mode_returns_mock_data_dir(self):
        """In dev mode, returns the mock data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dev_data_dir = Path(tmpdir) / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            with mock.patch.dict(
                os.environ, {"DISPATCH_DEV_DATA_DIR": str(dev_data_dir)}
            ):
                from dispatch_agents import get_data_dir

                result = get_data_dir()
                expected = dev_data_dir / "data"
                assert result == expected


# ============================================================================
# Test: _init_allowed_prefixes() function
# ============================================================================


class TestInitAllowedPrefixes:
    """Tests for _init_allowed_prefixes() function.

    Note: All prefixes are resolved to their canonical form to handle symlinks
    (e.g., macOS /var -> /private/var). Tests must resolve expected paths too.
    """

    def test_includes_data_dir(self):
        """Allowed prefixes include get_data_dir() (resolved)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dev_data_dir = Path(tmpdir) / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            with mock.patch.dict(
                os.environ, {"DISPATCH_DEV_DATA_DIR": str(dev_data_dir)}
            ):
                # Import the internal function
                from dispatch_agents import _init_allowed_prefixes, get_data_dir

                prefixes = _init_allowed_prefixes()
                # Resolve the path to match how prefixes are stored
                data_dir = str(get_data_dir().resolve())
                assert data_dir in prefixes

    def test_includes_tmp_directories(self):
        """Allowed prefixes include /tmp and other temp directories (resolved)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dev_data_dir = Path(tmpdir) / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            with mock.patch.dict(
                os.environ, {"DISPATCH_DEV_DATA_DIR": str(dev_data_dir)}
            ):
                from dispatch_agents import _init_allowed_prefixes

                prefixes = _init_allowed_prefixes()
                # All paths are resolved to canonical form (handles macOS symlinks)
                # e.g., /tmp -> /private/tmp, /var -> /private/var
                assert str(Path("/tmp").resolve()) in prefixes
                assert str(Path("/var/tmp").resolve()) in prefixes
                assert str(Path("/private/tmp").resolve()) in prefixes
                assert str(Path("/private/var/tmp").resolve()) in prefixes
                # System temp directory (handles macOS /var/folders/...)
                assert str(Path(tempfile.gettempdir()).resolve()) in prefixes

    def test_includes_agent_folder(self):
        """Allowed prefixes include the agent's folder (parent of dev-data, resolved)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate: /path/to/my-agent/.dispatch/dev-data
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            with mock.patch.dict(
                os.environ, {"DISPATCH_DEV_DATA_DIR": str(dev_data_dir)}
            ):
                from dispatch_agents import _init_allowed_prefixes

                prefixes = _init_allowed_prefixes()
                # Agent folder should be allowed (parent.parent of dev-data)
                # Resolve to handle symlinks (e.g., macOS /var/folders -> /private/var/folders)
                assert str(agent_folder.resolve()) in prefixes


# ============================================================================
# Test: DisallowedWriteError is exported
# ============================================================================


def test_disallowed_write_error_exported():
    """DisallowedWriteError is exported from dispatch_agents."""
    from dispatch_agents import DisallowedWriteError

    assert DisallowedWriteError is not None
    assert issubclass(DisallowedWriteError, Exception)

    # Can be instantiated with a message
    error = DisallowedWriteError("test message")
    assert "test message" in str(error)


# ============================================================================
# Test: Audit hook behavior (subprocess tests)
# ============================================================================


class TestAuditHookBehavior:
    """Tests for audit hook blocking writes.

    These tests run in subprocesses because audit hooks cannot be removed
    once installed, and the hook is installed at module import time.
    """

    def test_blocks_write_to_home_directory(self):
        """Audit hook blocks writes to home directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up dev mode environment
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            # Try to write to home directory (outside allowed locations)
            test_script = """
import os
import sys
from pathlib import Path

# This import installs the audit hook
from dispatch_agents import DisallowedWriteError

# Try to write to home directory
home_file = Path.home() / ".test-dispatch-isolation-file"
try:
    home_file.write_text("should not work")
    print("ERROR: Write succeeded when it should have been blocked")
    sys.exit(1)
except DisallowedWriteError as e:
    print(f"SUCCESS: Write blocked as expected: {e}")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: Unexpected exception: {type(e).__name__}: {e}")
    sys.exit(2)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_blocks_repeated_write_attempts(self):
        """Audit hook blocks repeated write attempts to the same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            # Try to write to the same disallowed path multiple times
            test_script = """
import sys
from pathlib import Path

from dispatch_agents import DisallowedWriteError

home_file = Path.home() / ".test-dispatch-repeated-file"

# First attempt - should be blocked with detailed message
try:
    home_file.write_text("attempt 1")
    print("ERROR: First write succeeded")
    sys.exit(1)
except DisallowedWriteError as e:
    if "outside allowed directories" not in str(e):
        print(f"ERROR: First error missing expected message: {e}")
        sys.exit(2)

# Second attempt - should ALSO be blocked (regression: was silently allowed)
try:
    home_file.write_text("attempt 2")
    print("ERROR: Second write succeeded when it should have been blocked")
    sys.exit(3)
except DisallowedWriteError as e:
    # Second error should have shorter message
    if "repeated attempt" not in str(e):
        print(f"ERROR: Second error missing 'repeated attempt': {e}")
        sys.exit(4)

# Third attempt - should still be blocked
try:
    home_file.write_text("attempt 3")
    print("ERROR: Third write succeeded")
    sys.exit(5)
except DisallowedWriteError:
    pass

print("SUCCESS: All repeated write attempts were blocked")
sys.exit(0)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_allows_write_to_data_dir(self):
        """Audit hook allows writes to get_data_dir()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            test_script = """
import sys
from dispatch_agents import get_data_dir

# Write to data directory
data_dir = get_data_dir()
data_dir.mkdir(parents=True, exist_ok=True)
test_file = data_dir / "test-file.txt"
try:
    test_file.write_text("test content")
    content = test_file.read_text()
    if content == "test content":
        print("SUCCESS: Write to data dir worked")
        sys.exit(0)
    else:
        print("ERROR: Content mismatch")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(2)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_allows_write_to_tmp(self):
        """Audit hook allows writes to /tmp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            test_script = """
import sys
import tempfile
from pathlib import Path

# Import to install audit hook
import dispatch_agents

# Write to temp directory
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    f.write("test content")
    temp_path = f.name

# Verify write worked
content = Path(temp_path).read_text()
if content == "test content":
    print("SUCCESS: Write to tmp worked")
    sys.exit(0)
else:
    print("ERROR: Content mismatch")
    sys.exit(1)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_allows_write_to_agent_folder(self):
        """Audit hook allows writes to agent's own folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            test_script = f"""
import sys
from pathlib import Path

# Import to install audit hook
import dispatch_agents

# Write to agent folder
agent_file = Path("{agent_folder}") / "test-file.txt"
try:
    agent_file.write_text("test content")
    content = agent_file.read_text()
    if content == "test content":
        print("SUCCESS: Write to agent folder worked")
        sys.exit(0)
    else:
        print("ERROR: Content mismatch")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(2)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_allow_arbitrary_writes_env_var(self):
        """DISPATCH_ALLOW_ARBITRARY_WRITES=1 disables the audit hook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            # Create a test file path outside allowed directories
            outside_path = Path(tmpdir) / "outside-agent" / "test.txt"

            test_script = f"""
import sys
from pathlib import Path

# Import to install audit hook (but it should be disabled)
import dispatch_agents

# Write outside allowed directories
outside_file = Path("{outside_path}")
outside_file.parent.mkdir(parents=True, exist_ok=True)
try:
    outside_file.write_text("test content")
    content = outside_file.read_text()
    if content == "test content":
        print("SUCCESS: Write worked with DISPATCH_ALLOW_ARBITRARY_WRITES")
        sys.exit(0)
    else:
        print("ERROR: Content mismatch")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(2)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "DISPATCH_ALLOW_ARBITRARY_WRITES": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_no_hook_in_production_mode(self):
        """Audit hook is not installed when DISPATCH_DEV_DATA_DIR is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write to arbitrary location (should work in production mode)
            test_file = Path(tmpdir) / "test-file.txt"

            test_script = f"""
import sys
from pathlib import Path

# Import - hook should NOT be installed (no DISPATCH_DEV_DATA_DIR)
import dispatch_agents

# Write anywhere
test_file = Path("{test_file}")
try:
    test_file.write_text("test content")
    content = test_file.read_text()
    if content == "test content":
        print("SUCCESS: Write worked in production mode")
        sys.exit(0)
    else:
        print("ERROR: Content mismatch")
        sys.exit(1)
except dispatch_agents.DisallowedWriteError as e:
    print(f"ERROR: Hook should not be installed: {{e}}")
    sys.exit(2)
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(3)
"""
            # Run without DISPATCH_DEV_DATA_DIR
            env = {k: v for k, v in os.environ.items() if k != "DISPATCH_DEV_DATA_DIR"}
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env=env,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout


# ============================================================================
# Test: Read operations are not blocked
# ============================================================================


class TestReadOperationsNotBlocked:
    """Ensure read operations are never blocked by the audit hook."""

    def test_can_read_files_anywhere(self):
        """Reading files from any location should always work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_folder = Path(tmpdir) / "my-agent"
            dev_data_dir = agent_folder / ".dispatch" / "dev-data"
            dev_data_dir.mkdir(parents=True)

            # Create a file outside allowed directories
            outside_file = Path(tmpdir) / "outside" / "test.txt"
            outside_file.parent.mkdir(parents=True)
            outside_file.write_text("readable content")

            test_script = f"""
import sys
from pathlib import Path

# Import to install audit hook
import dispatch_agents

# Read from outside allowed directories
outside_file = Path("{outside_file}")
try:
    content = outside_file.read_text()
    if content == "readable content":
        print("SUCCESS: Read from outside directories worked")
        sys.exit(0)
    else:
        print(f"ERROR: Content mismatch: {{content}}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(2)
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env={
                    **os.environ,
                    "DISPATCH_DEV_DATA_DIR": str(dev_data_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "SUCCESS" in result.stdout

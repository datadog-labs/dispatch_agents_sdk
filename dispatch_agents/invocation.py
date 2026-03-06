"""Shared types for function invocation across CLI and backend.

This module defines the core types used for direct function invocation (@fn decorator).
These types are shared between:
- SDK: Type hints for invoke() calls
- Backend: API request/response validation and database storage
- CLI: Local dev router API compatibility
"""

from enum import StrEnum

__all__ = ["InvocationStatus"]


class InvocationStatus(StrEnum):
    """Status of a function invocation.

    Tracks the lifecycle of a direct function call from initiation to completion.
    Used for polling-based status checks and database storage.
    """

    PENDING = "pending"  # Invocation created, not yet started
    RUNNING = "running"  # Invocation in progress
    COMPLETED = "completed"  # Invocation finished successfully
    ERROR = "error"  # Invocation failed with error

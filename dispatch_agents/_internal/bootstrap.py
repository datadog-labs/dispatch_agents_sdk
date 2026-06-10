"""SDK import-time bootstrap helpers."""

import sys
from pathlib import Path


def install_proto_import_path() -> None:
    """Add the SDK root so generated protobuf packages can be imported."""
    sdk_root = str(Path(__file__).parents[2])
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)

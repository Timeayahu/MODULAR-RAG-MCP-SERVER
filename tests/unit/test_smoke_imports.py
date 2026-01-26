"""
冒烟测试：验证关键模块可导入
"""

import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_smoke_imports():
    """确保关键模块可导入"""
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))

    import mcp_server  # noqa: F401
    import core  # noqa: F401
    import ingestion  # noqa: F401
    import libs  # noqa: F401
    import observability  # noqa: F401

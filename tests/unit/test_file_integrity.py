"""文件完整性检查测试。"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from libs.loader.file_integrity import (
    compute_sha256,
    mark_success,
    should_skip,
)


@pytest.mark.unit
def test_compute_sha256_consistent(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    first = compute_sha256(str(file_path))
    second = compute_sha256(str(file_path))
    assert first == second


@pytest.mark.unit
def test_mark_success_and_should_skip(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    file_hash = "abc123"

    assert should_skip(file_hash, str(cache_file)) is False
    mark_success(file_hash, str(cache_file))
    assert should_skip(file_hash, str(cache_file)) is True

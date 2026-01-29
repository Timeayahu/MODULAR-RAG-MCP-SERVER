"""文件完整性检查（SHA256）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


def compute_sha256(path: str) -> str:
    """计算文件 SHA256。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_cache(cache_path: Path) -> Dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache_path: Path, data: Dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def should_skip(file_hash: str, cache_path: str = "cache/processing/file_hashes.json") -> bool:
    """判断是否可跳过处理。"""
    cache_file = Path(cache_path)
    data = _load_cache(cache_file)
    return data.get(file_hash) == "success"


def mark_success(
    file_hash: str, cache_path: str = "cache/processing/file_hashes.json"
) -> None:
    """记录处理成功状态。"""
    cache_file = Path(cache_path)
    data = _load_cache(cache_file)
    data[file_hash] = "success"
    _save_cache(cache_file, data)

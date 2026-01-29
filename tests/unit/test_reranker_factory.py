"""Reranker 工厂测试。"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.none_reranker import NoneReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    """测试用 Fake Reranker。"""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace=None,  # type: ignore[override]
    ) -> List[Dict[str, Any]]:
        return list(reversed(candidates))


@pytest.mark.unit
def test_reranker_factory_routes_backend():
    """工厂应能根据 backend 创建对应实现。"""
    try:
        RerankerFactory.register("fake", FakeReranker)
        settings = load_settings(str(repo_root / "config" / "settings.yaml"))
        settings.rerank.backend = "fake"

        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, FakeReranker)
    finally:
        RerankerFactory.unregister("fake")


@pytest.mark.unit
def test_reranker_factory_unknown_backend():
    """未知 backend 应报错。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.rerank.backend = "unknown"

    with pytest.raises(ValueError) as excinfo:
        RerankerFactory.create(settings)

    assert "unknown" in str(excinfo.value)


@pytest.mark.unit
def test_none_reranker_keeps_order():
    """NoneReranker 应保持原顺序。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.rerank.backend = "none"

    reranker = RerankerFactory.create(settings)
    assert isinstance(reranker, NoneReranker)

    candidates = [{"id": 1}, {"id": 2}, {"id": 3}]
    results = reranker.rerank("q", candidates)
    assert results == candidates

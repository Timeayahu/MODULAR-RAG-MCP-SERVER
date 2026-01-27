"""Cross-Encoder Reranker 测试。"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeScorer:
    """测试用 scorer。"""

    def __init__(self, scores):
        self._scores = scores

    def score(self, query, candidates):
        return self._scores


@pytest.mark.unit
def test_cross_encoder_reranks_by_score():
    """应按分数降序排序。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.rerank.backend = "cross_encoder"
    settings.rerank.top_k = 2

    reranker = CrossEncoderReranker(settings.rerank, FakeScorer([0.2, 0.9, 0.1]))
    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    results = reranker.rerank("q", candidates)

    assert [item["id"] for item in results] == ["b", "a"]


@pytest.mark.unit
def test_cross_encoder_factory_creates_instance():
    """工厂应能创建 CrossEncoder Reranker。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.rerank.backend = "cross_encoder"

    reranker = RerankerFactory.create(settings)
    assert isinstance(reranker, CrossEncoderReranker)


@pytest.mark.unit
def test_cross_encoder_missing_scorer_raises():
    """未配置 scorer 应报错。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.rerank.backend = "cross_encoder"

    reranker = CrossEncoderReranker(settings.rerank, None)
    with pytest.raises(ValueError):
        reranker.rerank("q", [{"id": "a"}])

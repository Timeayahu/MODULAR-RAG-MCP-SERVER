"""Local Embedding 测试。"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.local_embedding import LocalEmbedding


@pytest.mark.unit
def test_local_embedding_returns_fixed_dim():
    """Local embedding 应返回固定维度向量。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.embedding.provider = "local"
    settings.embedding.dimensions = 8

    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed(["a", "b"])

    assert isinstance(embedding, LocalEmbedding)
    assert vectors == [[0.0] * 8, [0.0] * 8]


@pytest.mark.unit
def test_local_embedding_empty_input():
    """空输入应返回空列表。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.embedding.provider = "local"

    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed([])
    assert vectors == []

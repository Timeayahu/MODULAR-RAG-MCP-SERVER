"""Embedding provider 冒烟测试（mock HTTP）。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeResponse:
    """模拟 urllib 响应对象。"""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _mock_embedding_response(vectors):
    return {
        "data": [
            {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
        ]
    }


@pytest.mark.unit
@patch("urllib.request.urlopen")
def test_openai_embedding_returns_vectors(mock_urlopen):
    """OpenAI provider 应返回向量列表。"""
    mock_urlopen.return_value = FakeResponse(
        _mock_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    )
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.embedding.provider = "openai"
    settings.embedding.api_key = "test-key"
    settings.embedding.batch_size = 10

    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.unit
def test_openai_embedding_empty_input():
    """空输入应返回空列表。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.embedding.provider = "openai"
    settings.embedding.api_key = "test-key"

    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed([])
    assert vectors == []

"""LLM Reranker 测试。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM:
    """测试用 Fake LLM。"""

    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, messages):
        return self._response


@pytest.mark.unit
def test_llm_reranker_reorders_candidates():
    """LLM reranker 应按返回 id 顺序重排。"""
    prompt = "Query: {query}\nChunks:\n{chunks}\n"
    llm = FakeLLM(json.dumps(["b", "a"]))
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))

    reranker = LLMReranker(settings.rerank, llm, prompt_text=prompt)
    candidates = [
        {"id": "a", "text": "aaa"},
        {"id": "b", "text": "bbb"},
    ]

    results = reranker.rerank("q", candidates)
    assert [item["id"] for item in results] == ["b", "a"]


@pytest.mark.unit
def test_llm_reranker_invalid_output_raises():
    """非 JSON 输出应报错。"""
    prompt = "Query: {query}\nChunks:\n{chunks}\n"
    llm = FakeLLM("not-json")
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))

    reranker = LLMReranker(settings.rerank, llm, prompt_text=prompt)
    with pytest.raises(ValueError):
        reranker.rerank("q", [{"id": "a", "text": "aaa"}])


@pytest.mark.unit
def test_reranker_factory_llm_backend_creates_instance():
    """backend=llm 时工厂应创建 LLMReranker。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.rerank.backend = "llm"

    with patch("libs.reranker.llm_reranker.LLMFactory.create") as mocked:
        mocked.return_value = FakeLLM(json.dumps(["a"]))
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, LLMReranker)

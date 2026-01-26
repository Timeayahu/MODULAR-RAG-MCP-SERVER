"""LLM 工厂测试。"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from typing import Dict, List

from core.settings import load_settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    """测试用 Fake LLM。"""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return "ok"


@pytest.mark.unit
def test_llm_factory_routes_provider():
    """工厂应能根据 provider 创建对应实现。"""
    try:
        LLMFactory.register("fake", FakeLLM)
        settings = load_settings(str(repo_root / "config" / "settings.yaml"))
        settings.llm.provider = "fake"

        llm = LLMFactory.create(settings)
        assert isinstance(llm, FakeLLM)
    finally:
        LLMFactory.unregister("fake")


@pytest.mark.unit
def test_llm_factory_unknown_provider():
    """未知 provider 应报错。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.llm.provider = "unknown"

    with pytest.raises(ValueError) as excinfo:
        LLMFactory.create(settings)

    assert "unknown" in str(excinfo.value)

"""LLM provider 冒烟测试（mock HTTP）。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.llm.llm_factory import LLMFactory


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


def _mock_response_content(text: str = "ok") -> dict:
    return {"choices": [{"message": {"content": text}}]}


@pytest.mark.unit
@patch("urllib.request.urlopen")
def test_openai_llm_chat_returns_content(mock_urlopen):
    """OpenAI provider 应返回 content。"""
    mock_urlopen.return_value = FakeResponse(_mock_response_content("hello"))
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.llm.provider = "openai"
    settings.llm.api_key = "test-key"

    llm = LLMFactory.create(settings)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "hello"


@pytest.mark.unit
@patch("urllib.request.urlopen")
def test_azure_llm_chat_returns_content(mock_urlopen):
    """Azure provider 应返回 content。"""
    mock_urlopen.return_value = FakeResponse(_mock_response_content("azure-ok"))
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.llm.provider = "azure"
    settings.llm.api_key = "test-key"
    settings.llm.base_url = "https://example.azure.com/openai/deployments/test"

    llm = LLMFactory.create(settings)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "azure-ok"


@pytest.mark.unit
@patch("urllib.request.urlopen")
def test_deepseek_llm_chat_returns_content(mock_urlopen):
    """DeepSeek provider 应返回 content。"""
    mock_urlopen.return_value = FakeResponse(_mock_response_content("deepseek-ok"))
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.llm.provider = "deepseek"
    settings.llm.api_key = "test-key"

    llm = LLMFactory.create(settings)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "deepseek-ok"


@pytest.mark.unit
@patch("urllib.request.urlopen")
def test_ollama_llm_chat_returns_content(mock_urlopen):
    """Ollama provider 应返回 content。"""
    mock_urlopen.return_value = FakeResponse(_mock_response_content("ollama-ok"))
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.llm.provider = "ollama"
    settings.llm.base_url = "http://localhost:11434"

    llm = LLMFactory.create(settings)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "ollama-ok"


@pytest.mark.unit
def test_llm_chat_invalid_messages():
    """消息格式错误应报错。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.llm.provider = "openai"
    settings.llm.api_key = "test-key"

    llm = LLMFactory.create(settings)
    with pytest.raises(ValueError) as excinfo:
        llm.chat([{"role": "user"}])  # missing content

    assert "openai" in str(excinfo.value)

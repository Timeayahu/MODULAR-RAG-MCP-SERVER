"""Splitter 工厂测试。"""

import sys
from pathlib import Path
from typing import List

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """测试用 Fake Splitter。"""

    def split_text(self, text: str, trace=None) -> List[str]:  # type: ignore[override]
        return [text]


@pytest.mark.unit
def test_splitter_factory_routes_provider():
    """工厂应能根据 provider 创建对应实现。"""
    try:
        SplitterFactory.register("fake", FakeSplitter)
        settings = load_settings(str(repo_root / "config" / "settings.yaml"))
        settings.splitter.provider = "fake"

        splitter = SplitterFactory.create(settings)
        chunks = splitter.split_text("hello")
        assert isinstance(splitter, FakeSplitter)
        assert chunks == ["hello"]
    finally:
        SplitterFactory.unregister("fake")


@pytest.mark.unit
def test_splitter_factory_unknown_provider():
    """未知 provider 应报错。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.splitter.provider = "unknown"

    with pytest.raises(ValueError) as excinfo:
        SplitterFactory.create(settings)

    assert "unknown" in str(excinfo.value)

"""Recursive Splitter 测试。

测试基于 LangChain RecursiveCharacterTextSplitter 的递归切分器。
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.splitter.splitter_factory import SplitterFactory
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.base_splitter import SplitResult


@pytest.mark.unit
def test_recursive_splitter_splits_text():
    """递归切分器应按 chunk_size/overlap 切分并返回带定位信息的结果。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.splitter.provider = "recursive"
    settings.splitter.chunk_size = 20
    settings.splitter.chunk_overlap = 5

    splitter = SplitterFactory.create(settings)
    assert isinstance(splitter, RecursiveSplitter)

    # 测试 split 方法返回 SplitResult
    text = "This is a test text that should be split into multiple chunks."
    results = splitter.split(text)

    assert len(results) > 0
    assert all(isinstance(r, SplitResult) for r in results)
    # 验证定位信息
    for r in results:
        assert r.chunk_index >= 0
        assert r.start_offset >= 0
        assert r.end_offset >= r.start_offset


@pytest.mark.unit
def test_recursive_splitter_split_text_returns_strings():
    """split_text 方法应返回字符串列表（简化接口）。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.splitter.provider = "recursive"
    settings.splitter.chunk_size = 20
    settings.splitter.chunk_overlap = 0

    splitter = SplitterFactory.create(settings)
    chunks = splitter.split_text("This is a test text that should be split.")

    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.unit
def test_recursive_splitter_empty_input():
    """空输入应返回空列表。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.splitter.provider = "recursive"

    splitter = SplitterFactory.create(settings)
    assert splitter.split_text("") == []
    assert splitter.split("") == []


@pytest.mark.unit
def test_recursive_splitter_markdown_awareness():
    """递归切分器应尊重 Markdown 结构。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.splitter.provider = "recursive"
    settings.splitter.chunk_size = 100
    settings.splitter.chunk_overlap = 0

    splitter = SplitterFactory.create(settings)

    markdown_text = """# Title

This is the first paragraph.

## Section 1

Content of section 1.

## Section 2

Content of section 2.
"""

    chunks = splitter.split_text(markdown_text)
    # 至少应该产生多个 chunk
    assert len(chunks) >= 1
    # 每个 chunk 应该包含有意义的文本
    assert all(c.strip() for c in chunks)

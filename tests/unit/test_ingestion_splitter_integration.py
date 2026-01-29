"""Ingestion Pipeline 与 Splitter 的集成测试（阶段 C4）。

验证要点：
- IngestionPipeline 能使用 PdfLoader 加载文本文件；
- 通过 SplitterFactory 创建的 Splitter 能正确切分文本；
- 修改 splitter.chunk_size 会改变切分结果（chunk 数量或长度）；
- run_loader_and_splitter 返回 Chunk 对象列表（带定位信息）。
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import (  # noqa: E402
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    ObservabilityConfig,
    RetrievalConfig,
    RerankConfig,
    Settings,
    SplitterConfig,
    VectorStoreConfig,
)
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from ingestion.models import Chunk  # noqa: E402


def _build_minimal_settings(chunk_size: int) -> Settings:
    """构造最小可用 Settings，便于单元测试。

    仅关注 Splitter 相关字段，其余字段使用占位默认值。
    """

    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterConfig(provider="recursive", chunk_size=chunk_size, chunk_overlap=0),
        vector_store=VectorStoreConfig(provider="chroma"),
        retrieval=RetrievalConfig(),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(),
        observability=ObservabilityConfig(),
    )


@pytest.mark.unit
def test_ingestion_pipeline_splitter_respects_chunk_size(tmp_path: Path):
    """测试 Pipeline 的切分行为受 chunk_size 配置影响。"""
    # 准备一个足够长的"伪 PDF" 文本文件
    sample_file = tmp_path / "sample.pdf"
    # 100 个字符的简单文本
    base_text = "abcdefghij " * 10  # 添加空格以便更好地切分
    sample_file.write_text(base_text, encoding="utf-8")

    # 配置 1：较小 chunk_size
    small_settings = _build_minimal_settings(chunk_size=20)
    small_pipeline = IngestionPipeline(small_settings)
    small_chunks = small_pipeline.run_loader_and_splitter(str(sample_file))

    # 配置 2：较大 chunk_size
    large_settings = _build_minimal_settings(chunk_size=60)
    large_pipeline = IngestionPipeline(large_settings)
    large_chunks = large_pipeline.run_loader_and_splitter(str(sample_file))

    # 验证返回类型为 Chunk 列表
    assert all(isinstance(c, Chunk) for c in small_chunks)
    assert all(isinstance(c, Chunk) for c in large_chunks)

    # 验证：较小 chunk_size 会产生更多的 chunk
    assert len(small_chunks) > len(large_chunks)


@pytest.mark.unit
def test_ingestion_pipeline_chunks_have_position_info(tmp_path: Path):
    """测试 Chunk 包含正确的定位信息。"""
    sample_file = tmp_path / "test.pdf"
    sample_file.write_text("This is a test document with some content.", encoding="utf-8")

    settings = _build_minimal_settings(chunk_size=20)
    pipeline = IngestionPipeline(settings)
    chunks = pipeline.run_loader_and_splitter(str(sample_file))

    # 验证每个 Chunk 都有定位信息
    for chunk in chunks:
        assert chunk.chunk_index >= 0
        assert chunk.start_offset >= 0
        assert chunk.end_offset >= chunk.start_offset
        assert chunk.source_doc_id is not None


@pytest.mark.unit
def test_ingestion_pipeline_split_document_text(tmp_path: Path):
    """测试 split_document_text 方法返回字符串列表。"""
    sample_file = tmp_path / "test.pdf"
    sample_file.write_text("Test content for splitting.", encoding="utf-8")

    settings = _build_minimal_settings(chunk_size=10)
    pipeline = IngestionPipeline(settings)

    doc = pipeline.load_document(str(sample_file))
    text_chunks = pipeline.split_document_text(doc)

    # 验证返回字符串列表
    assert isinstance(text_chunks, list)
    assert all(isinstance(c, str) for c in text_chunks)

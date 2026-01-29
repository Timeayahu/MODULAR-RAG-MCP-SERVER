"""Ingestion 数据模型测试。

测试与 LlamaIndex Document/TextNode 兼容的数据模型。
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.models import Chunk, Document


@pytest.mark.unit
def test_document_to_dict():
    """Document 应正确序列化为 dict。"""
    doc = Document(id="doc-1", text="hello", metadata={"source": "unit"})
    payload = doc.to_dict()
    assert payload["id"] == "doc-1"
    assert payload["text"] == "hello"
    assert payload["metadata"]["source"] == "unit"


@pytest.mark.unit
def test_document_with_images():
    """Document 应支持 images 字段。"""
    doc = Document(
        id="doc-2",
        text="content with image",
        metadata={"source": "test"},
        images=[{"image_id": "img_1", "url": "test.png"}],
    )
    payload = doc.to_dict()
    assert len(payload["images"]) == 1
    assert payload["images"][0]["image_id"] == "img_1"


@pytest.mark.unit
def test_chunk_to_dict():
    """Chunk 应正确序列化为 dict。"""
    chunk = Chunk(
        id="chunk-1",
        text="text",
        metadata={"page": 1},
        start_offset=0,
        end_offset=4,
    )
    payload = chunk.to_dict()
    assert payload["id"] == "chunk-1"
    assert payload["start_offset"] == 0
    assert payload["end_offset"] == 4


@pytest.mark.unit
def test_chunk_with_full_metadata():
    """Chunk 应支持完整的元数据字段。"""
    chunk = Chunk(
        id="chunk-2",
        text="chunk text",
        metadata={"key": "value"},
        source_doc_id="doc-1",
        chunk_index=3,
        start_offset=100,
        end_offset=200,
        image_refs=["img_1", "img_2"],
    )
    payload = chunk.to_dict()
    assert payload["source_doc_id"] == "doc-1"
    assert payload["chunk_index"] == 3
    assert payload["image_refs"] == ["img_1", "img_2"]


@pytest.mark.unit
def test_document_llama_index_conversion():
    """测试 Document 与 LlamaIndex Document 的转换。"""
    try:
        from llama_index.core.schema import Document as LlamaDocument
        LLAMA_INDEX_AVAILABLE = True
    except ImportError:
        LLAMA_INDEX_AVAILABLE = False

    if not LLAMA_INDEX_AVAILABLE:
        pytest.skip("llama-index 未安装")

    # 创建 Document
    doc = Document(
        id="test-doc",
        text="Test content",
        metadata={"source": "test", "page": 1},
        images=[{"image_id": "img_0", "url": "image.png"}],
    )

    # 转换为 LlamaIndex Document
    llama_doc = doc.to_llama_document()
    assert llama_doc.doc_id == "test-doc"
    assert llama_doc.text == "Test content"
    assert llama_doc.metadata["source"] == "test"

    # 从 LlamaIndex Document 转回
    doc_back = Document.from_llama_document(llama_doc)
    assert doc_back.id == "test-doc"
    assert doc_back.text == "Test content"
    assert doc_back.metadata["source"] == "test"


@pytest.mark.unit
def test_chunk_llama_index_conversion():
    """测试 Chunk 与 LlamaIndex TextNode 的转换。"""
    try:
        from llama_index.core.schema import TextNode
        LLAMA_INDEX_AVAILABLE = True
    except ImportError:
        LLAMA_INDEX_AVAILABLE = False

    if not LLAMA_INDEX_AVAILABLE:
        pytest.skip("llama-index 未安装")

    # 创建 Chunk
    chunk = Chunk(
        id="test-chunk",
        text="Chunk content",
        metadata={"key": "value"},
        source_doc_id="parent-doc",
        chunk_index=5,
        start_offset=100,
        end_offset=200,
        image_refs=["img_1"],
    )

    # 转换为 LlamaIndex TextNode
    node = chunk.to_llama_node()
    assert node.id_ == "test-chunk"
    assert node.text == "Chunk content"
    assert node.start_char_idx == 100
    assert node.end_char_idx == 200

    # 从 LlamaIndex TextNode 转回
    chunk_back = Chunk.from_llama_node(node)
    assert chunk_back.id == "test-chunk"
    assert chunk_back.text == "Chunk content"
    assert chunk_back.start_offset == 100
    assert chunk_back.end_offset == 200

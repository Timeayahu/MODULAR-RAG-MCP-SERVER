"""VectorUpserter 测试 - 重点测试幂等性。

根据 DEV_SPEC C11 验收标准：
- 同一 chunk 两次 upsert 产生相同 id
- 内容变更 id 变更
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.models import Chunk
from ingestion.storage.vector_upserter import VectorUpserter


@pytest.fixture
def mock_vector_store():
    """创建 Mock VectorStore。"""
    store = Mock()
    store.upsert.return_value = None
    return store


@pytest.fixture
def sample_chunk():
    """创建示例 Chunk。"""
    return Chunk(
        id="original-id",  # 原始 id 会被重新生成
        text="This is test content for chunk",
        metadata={"source": "doc1.pdf", "page": 1},
        source_doc_id="doc1.pdf",
        chunk_index=0,
    )


@pytest.mark.unit
def test_vector_upserter_generates_stable_id(mock_vector_store, sample_chunk):
    """验收标准：同一 chunk 两次 upsert 产生相同 id。"""
    upserter = VectorUpserter(mock_vector_store)

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term1": 1.0}

    # 第一次 upsert
    chunk_ids_1 = upserter.upsert(
        [sample_chunk],
        [dense_vec],
        [sparse_vec],
    )

    # 第二次 upsert（相同内容）
    chunk_ids_2 = upserter.upsert(
        [sample_chunk],
        [dense_vec],
        [sparse_vec],
    )

    # 应生成相同的 chunk_id
    assert chunk_ids_1 == chunk_ids_2
    assert len(chunk_ids_1) == 1


@pytest.mark.unit
def test_vector_upserter_id_changes_with_content(mock_vector_store):
    """验收标准：内容变更 id 变更。"""
    upserter = VectorUpserter(mock_vector_store)

    # 创建两个内容不同的 chunk
    chunk1 = Chunk(
        id="id1",
        text="Original content",
        metadata={"source": "doc1.pdf"},
        source_doc_id="doc1.pdf",
    )

    chunk2 = Chunk(
        id="id2",
        text="Modified content",  # 内容变更
        metadata={"source": "doc1.pdf"},
        source_doc_id="doc1.pdf",
    )

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    # Upsert 两个 chunk
    chunk_id_1 = upserter.upsert_single(chunk1, dense_vec, sparse_vec)
    chunk_id_2 = upserter.upsert_single(chunk2, dense_vec, sparse_vec)

    # ID 应该不同
    assert chunk_id_1 != chunk_id_2


@pytest.mark.unit
def test_vector_upserter_id_stable_across_instances(sample_chunk):
    """相同内容在不同 upserter 实例中产生相同 id。"""
    store1 = Mock()
    store1.upsert.return_value = None
    upserter1 = VectorUpserter(store1)

    store2 = Mock()
    store2.upsert.return_value = None
    upserter2 = VectorUpserter(store2)

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    # 在两个不同的 upserter 实例中 upsert
    chunk_id_1 = upserter1.upsert_single(sample_chunk, dense_vec, sparse_vec)
    chunk_id_2 = upserter2.upsert_single(sample_chunk, dense_vec, sparse_vec)

    # 应产生相同的 id
    assert chunk_id_1 == chunk_id_2


@pytest.mark.unit
def test_vector_upserter_batch_upsert(mock_vector_store):
    """测试批量 upsert。"""
    upserter = VectorUpserter(mock_vector_store)

    chunks = [
        Chunk(id=f"id{i}", text=f"content {i}", metadata={}, source_doc_id=f"doc{i}")
        for i in range(3)
    ]

    dense_vecs = [[float(i)] * 3 for i in range(3)]
    sparse_vecs = [{f"term{i}": 1.0} for i in range(3)]

    chunk_ids = upserter.upsert(chunks, dense_vecs, sparse_vecs)

    # 应返回 3 个 chunk_id
    assert len(chunk_ids) == 3

    # 所有 id 应该不同
    assert len(set(chunk_ids)) == 3

    # 应调用 vector_store.upsert 一次
    assert mock_vector_store.upsert.call_count == 1


@pytest.mark.unit
def test_vector_upserter_empty_chunks_raises_error(mock_vector_store):
    """空 chunks 列表应抛出错误。"""
    upserter = VectorUpserter(mock_vector_store)

    with pytest.raises(ValueError, match="chunks 不能为空"):
        upserter.upsert([], [], [])


@pytest.mark.unit
def test_vector_upserter_length_mismatch_raises_error(mock_vector_store, sample_chunk):
    """长度不匹配应抛出错误。"""
    upserter = VectorUpserter(mock_vector_store)

    chunks = [sample_chunk]
    dense_vecs = [[0.1, 0.2], [0.3, 0.4]]  # 长度不匹配
    sparse_vecs = [{"term": 1.0}]

    with pytest.raises(ValueError, match="dense_vectors 数量.*不一致"):
        upserter.upsert(chunks, dense_vecs, sparse_vecs)


@pytest.mark.unit
def test_vector_upserter_includes_metadata(mock_vector_store, sample_chunk):
    """验证存储的记录包含完整的 metadata。"""
    upserter = VectorUpserter(mock_vector_store)

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    upserter.upsert([sample_chunk], [dense_vec], [sparse_vec])

    # 获取调用参数
    call_args = mock_vector_store.upsert.call_args
    records = call_args[0][0]

    assert len(records) == 1
    record = records[0]

    # 验证记录结构（与 BaseVectorStore 契约对齐）
    assert "id" in record
    assert "vector" in record
    assert "text" in record
    assert "metadata" in record

    # 验证内容
    assert record["text"] == sample_chunk.text
    assert record["vector"] == dense_vec
    assert record["metadata"]["sparse_vector"] == sparse_vec

    # 验证 metadata 包含必要信息
    assert record["metadata"]["source"] == "doc1.pdf"
    assert record["metadata"]["chunk_index"] == 0
    assert record["metadata"]["source_doc_id"] == "doc1.pdf"


@pytest.mark.unit
def test_vector_upserter_id_includes_source_path(mock_vector_store):
    """测试 chunk_id 包含 source_path 信息。"""
    upserter = VectorUpserter(mock_vector_store)

    # 相同内容但不同 source
    chunk1 = Chunk(
        id="id1",
        text="same content",
        metadata={"source": "doc1.pdf"},
        source_doc_id="doc1.pdf",
    )

    chunk2 = Chunk(
        id="id2",
        text="same content",
        metadata={"source": "doc2.pdf"},
        source_doc_id="doc2.pdf",
    )

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    # 即使内容相同，不同 source 应产生不同 id
    chunk_id_1 = upserter.upsert_single(chunk1, dense_vec, sparse_vec)
    chunk_id_2 = upserter.upsert_single(chunk2, dense_vec, sparse_vec)

    assert chunk_id_1 != chunk_id_2


@pytest.mark.unit
def test_vector_upserter_id_includes_chunk_index(mock_vector_store):
    """测试 chunk_id 包含 chunk_index 信息。"""
    upserter = VectorUpserter(mock_vector_store)

    # 相同内容但不同 chunk_index
    chunk1 = Chunk(
        id="id1",
        text="same content",
        metadata={"source": "doc1.pdf"},
        source_doc_id="doc1.pdf",
        chunk_index=0,
    )

    chunk2 = Chunk(
        id="id2",
        text="same content",
        metadata={"source": "doc1.pdf"},
        source_doc_id="doc1.pdf",
        chunk_index=1,
    )

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    # 不同 chunk_index 应产生不同 id
    chunk_id_1 = upserter.upsert_single(chunk1, dense_vec, sparse_vec)
    chunk_id_2 = upserter.upsert_single(chunk2, dense_vec, sparse_vec)

    assert chunk_id_1 != chunk_id_2


@pytest.mark.unit
def test_vector_upserter_with_image_refs(mock_vector_store):
    """测试包含图片引用的 chunk。"""
    upserter = VectorUpserter(mock_vector_store)

    chunk = Chunk(
        id="id1",
        text="chunk with images",
        metadata={"source": "doc1.pdf"},
        source_doc_id="doc1.pdf",
        image_refs=["img_1", "img_2"],
    )

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    upserter.upsert_single(chunk, dense_vec, sparse_vec)

    # 验证 image_refs 被包含在 metadata 中
    call_args = mock_vector_store.upsert.call_args
    records = call_args[0][0]
    assert records[0]["metadata"]["image_refs"] == ["img_1", "img_2"]


@pytest.mark.unit
def test_vector_upserter_with_trace_context(mock_vector_store, sample_chunk):
    """测试传递 trace 上下文。"""
    upserter = VectorUpserter(mock_vector_store)
    mock_trace = Mock()

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    upserter.upsert([sample_chunk], [dense_vec], [sparse_vec], trace=mock_trace)

    # 验证 trace 被传递给 vector_store
    call_kwargs = mock_vector_store.upsert.call_args[1]
    assert call_kwargs.get("trace") == mock_trace


@pytest.mark.unit
def test_vector_upserter_propagates_store_error(mock_vector_store, sample_chunk):
    """测试传播 vector_store 错误。"""
    mock_vector_store.upsert.side_effect = RuntimeError("Storage Error")
    upserter = VectorUpserter(mock_vector_store)

    dense_vec = [0.1, 0.2, 0.3]
    sparse_vec = {"term": 1.0}

    with pytest.raises(RuntimeError, match="Storage Error"):
        upserter.upsert([sample_chunk], [dense_vec], [sparse_vec])

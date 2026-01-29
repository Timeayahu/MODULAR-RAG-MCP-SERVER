"""DenseEncoder 测试。

根据 DEV_SPEC C8 验收标准：
- encoder 输出向量数量与 chunks 数量一致
- 输出向量维度一致
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.models import Chunk


@pytest.fixture
def mock_embedding():
    """创建 Mock Embedding。"""
    embedding = Mock()
    # 默认返回固定维度的向量
    embedding.embed.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    return embedding


@pytest.fixture
def sample_chunks():
    """创建示例 Chunks。"""
    return [
        Chunk(
            id="chunk-1",
            text="First chunk text",
            metadata={"source": "doc1.pdf"},
        ),
        Chunk(
            id="chunk-2",
            text="Second chunk text",
            metadata={"source": "doc1.pdf"},
        ),
        Chunk(
            id="chunk-3",
            text="Third chunk text",
            metadata={"source": "doc1.pdf"},
        ),
    ]


@pytest.mark.unit
def test_dense_encoder_basic(mock_embedding, sample_chunks):
    """测试基本的向量编码功能。"""
    encoder = DenseEncoder(mock_embedding)

    vectors = encoder.encode(sample_chunks)

    # 验证调用了 embedding
    assert mock_embedding.embed.called
    called_texts = mock_embedding.embed.call_args[0][0]
    assert len(called_texts) == 3
    assert called_texts[0] == "First chunk text"

    # 验收标准：向量数量与 chunks 数量一致
    assert len(vectors) == len(sample_chunks)

    # 验收标准：向量维度一致
    assert all(len(v) == 3 for v in vectors)


@pytest.mark.unit
def test_dense_encoder_empty_chunks_raises_error(mock_embedding):
    """空 chunks 列表应抛出错误。"""
    encoder = DenseEncoder(mock_embedding)

    with pytest.raises(ValueError, match="chunks 不能为空"):
        encoder.encode([])


@pytest.mark.unit
def test_dense_encoder_single_chunk(mock_embedding):
    """测试编码单个 chunk。"""
    encoder = DenseEncoder(mock_embedding)
    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]

    chunk = Chunk(
        id="chunk-1",
        text="Single chunk",
        metadata={},
    )

    vector = encoder.encode_single(chunk)

    # 应返回一个向量（不是列表的列表）
    assert isinstance(vector, list)
    assert len(vector) == 3
    assert vector == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_dense_encoder_handles_empty_text(mock_embedding):
    """测试处理包含空文本的 chunks。"""
    encoder = DenseEncoder(mock_embedding)
    mock_embedding.embed.return_value = [
        [0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0],  # 空文本可能返回零向量
    ]

    chunks = [
        Chunk(id="chunk-1", text="Normal text", metadata={}),
        Chunk(id="chunk-2", text="", metadata={}),  # 空文本
    ]

    vectors = encoder.encode(chunks)

    # 应正常处理，不抛出异常
    assert len(vectors) == 2


@pytest.mark.unit
def test_dense_encoder_preserves_order(mock_embedding, sample_chunks):
    """测试保持 chunks 顺序。"""
    encoder = DenseEncoder(mock_embedding)
    mock_embedding.embed.return_value = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ]

    vectors = encoder.encode(sample_chunks)

    # 验证顺序保持一致
    assert vectors[0] == [1.0, 1.0, 1.0]
    assert vectors[1] == [2.0, 2.0, 2.0]
    assert vectors[2] == [3.0, 3.0, 3.0]


@pytest.mark.unit
def test_dense_encoder_embedding_failure_propagates(mock_embedding, sample_chunks):
    """测试 embedding 失败时传播异常。"""
    encoder = DenseEncoder(mock_embedding)
    mock_embedding.embed.side_effect = RuntimeError("API Error")

    with pytest.raises(RuntimeError, match="API Error"):
        encoder.encode(sample_chunks)


@pytest.mark.unit
def test_dense_encoder_validates_vector_count(mock_embedding, sample_chunks):
    """测试验证向量数量与 chunks 数量一致。"""
    encoder = DenseEncoder(mock_embedding)
    # 故意返回错误数量的向量
    mock_embedding.embed.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        # 缺少第三个向量
    ]

    with pytest.raises(RuntimeError, match="向量数量.*不一致"):
        encoder.encode(sample_chunks)


@pytest.mark.unit
def test_dense_encoder_with_trace_context(mock_embedding, sample_chunks):
    """测试传递 trace 上下文。"""
    encoder = DenseEncoder(mock_embedding)
    mock_trace = Mock()

    encoder.encode(sample_chunks, trace=mock_trace)

    # 验证 trace 被传递给 embedding
    call_kwargs = mock_embedding.embed.call_args[1]
    assert call_kwargs.get("trace") == mock_trace


@pytest.mark.unit
def test_dense_encoder_embedding_dim_property():
    """测试 embedding_dim 属性。"""
    # 测试没有 config 的情况
    mock_embedding_no_config = Mock(spec=[])  # 没有任何属性
    encoder = DenseEncoder(mock_embedding_no_config)
    dim = encoder.embedding_dim
    assert dim is None

    # 测试有 config 但没有 dimension 的情况
    mock_embedding_no_dim = Mock()
    mock_embedding_no_dim.config = Mock(spec=[])  # config 没有 dimension
    encoder2 = DenseEncoder(mock_embedding_no_dim)
    assert encoder2.embedding_dim is None

    # 测试有 config 且有 dimension 的情况
    mock_embedding_with_dim = Mock()
    mock_config = Mock()
    mock_config.dimension = 384
    mock_embedding_with_dim.config = mock_config
    encoder3 = DenseEncoder(mock_embedding_with_dim)
    assert encoder3.embedding_dim == 384


@pytest.mark.unit
def test_dense_encoder_large_batch(mock_embedding):
    """测试处理大批量 chunks。"""
    encoder = DenseEncoder(mock_embedding)

    # 创建 100 个 chunks
    chunks = [
        Chunk(id=f"chunk-{i}", text=f"Text {i}", metadata={})
        for i in range(100)
    ]

    # Mock 返回 100 个向量
    mock_embedding.embed.return_value = [
        [float(i), float(i), float(i)] for i in range(100)
    ]

    vectors = encoder.encode(chunks)

    # 验收标准：向量数量与 chunks 数量一致
    assert len(vectors) == 100

    # 验收标准：向量维度一致
    assert all(len(v) == 3 for v in vectors)

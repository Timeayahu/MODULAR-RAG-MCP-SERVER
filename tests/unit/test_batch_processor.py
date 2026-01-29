"""BatchProcessor 测试。

根据 DEV_SPEC C10 验收标准：
- batch_size=2 时对 5 chunks 分成 3 批，且顺序稳定
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.models import Chunk


@pytest.fixture
def sample_chunks():
    """创建示例 Chunks。"""
    return [
        Chunk(id=f"chunk-{i}", text=f"Text {i}", metadata={})
        for i in range(5)
    ]


@pytest.fixture
def mock_dense_encoder():
    """创建 Mock DenseEncoder。"""
    encoder = Mock()
    # 每次调用返回对应数量的向量
    encoder.encode.side_effect = lambda chunks, trace=None: [
        [float(i), float(i), float(i)] for i in range(len(chunks))
    ]
    return encoder


@pytest.fixture
def mock_sparse_encoder():
    """创建 Mock SparseEncoder。"""
    encoder = Mock()
    # 每次调用返回对应数量的稀疏向量
    encoder.encode.side_effect = lambda chunks, trace=None: [
        {f"term_{i}": float(i)} for i in range(len(chunks))
    ]
    return encoder


@pytest.mark.unit
def test_batch_processor_splits_into_batches(mock_dense_encoder, sample_chunks):
    """验收标准：batch_size=2 时对 5 chunks 分成 3 批。"""
    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        batch_size=2,
    )

    dense_vectors, _ = processor.process(sample_chunks)

    # 验证调用次数（5个chunks，batch_size=2，应该分3批）
    assert mock_dense_encoder.encode.call_count == 3

    # 验证每批的大小
    call_args_list = mock_dense_encoder.encode.call_args_list
    assert len(call_args_list[0][0][0]) == 2  # 第1批：2个chunks
    assert len(call_args_list[1][0][0]) == 2  # 第2批：2个chunks
    assert len(call_args_list[2][0][0]) == 1  # 第3批：1个chunk

    # 验证输出总数
    assert len(dense_vectors) == 5


@pytest.mark.unit
def test_batch_processor_maintains_order(mock_dense_encoder, mock_sparse_encoder):
    """验收标准：顺序稳定。"""
    # 创建带有明确顺序标识的 chunks
    chunks = [
        Chunk(id=f"chunk-{i}", text=f"Order {i}", metadata={"order": i})
        for i in range(10)
    ]

    # Mock encoder 返回与输入顺序相关的结果
    mock_dense_encoder.encode.side_effect = lambda cks, trace=None: [
        [float(int(ck.id.split("-")[1]))] for ck in cks
    ]
    mock_sparse_encoder.encode.side_effect = lambda cks, trace=None: [
        {"id": int(ck.id.split("-")[1])} for ck in cks
    ]

    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        sparse_encoder=mock_sparse_encoder,
        batch_size=3,
    )

    dense_vectors, sparse_vectors = processor.process(chunks)

    # 验证顺序保持
    for i in range(10):
        assert dense_vectors[i] == [float(i)]
        assert sparse_vectors[i] == {"id": i}


@pytest.mark.unit
def test_batch_processor_empty_chunks_raises_error(mock_dense_encoder):
    """空 chunks 列表应抛出错误。"""
    processor = BatchProcessor(dense_encoder=mock_dense_encoder)

    with pytest.raises(ValueError, match="chunks 不能为空"):
        processor.process([])


@pytest.mark.unit
def test_batch_processor_no_encoder_raises_error():
    """没有配置任何 encoder 应抛出错误。"""
    processor = BatchProcessor()  # 没有 encoder

    chunks = [Chunk(id="1", text="test", metadata={})]

    with pytest.raises(ValueError, match="至少需要配置一个 encoder"):
        processor.process(chunks)


@pytest.mark.unit
def test_batch_processor_invalid_batch_size():
    """无效的 batch_size 应抛出错误。"""
    with pytest.raises(ValueError, match="batch_size 必须大于 0"):
        BatchProcessor(batch_size=0)

    with pytest.raises(ValueError, match="batch_size 必须大于 0"):
        BatchProcessor(batch_size=-1)


@pytest.mark.unit
def test_batch_processor_dense_only(mock_dense_encoder, sample_chunks):
    """测试仅 dense 编码。"""
    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        batch_size=2,
    )

    dense_vectors = processor.process_dense_only(sample_chunks)

    # 应调用 dense encoder
    assert mock_dense_encoder.encode.called

    # 应返回正确数量的向量
    assert len(dense_vectors) == 5


@pytest.mark.unit
def test_batch_processor_sparse_only(mock_sparse_encoder, sample_chunks):
    """测试仅 sparse 编码。"""
    processor = BatchProcessor(
        sparse_encoder=mock_sparse_encoder,
        batch_size=2,
    )

    sparse_vectors = processor.process_sparse_only(sample_chunks)

    # 应调用 sparse encoder
    assert mock_sparse_encoder.encode.called

    # 应返回正确数量的向量
    assert len(sparse_vectors) == 5


@pytest.mark.unit
def test_batch_processor_both_encoders(
    mock_dense_encoder, mock_sparse_encoder, sample_chunks
):
    """测试同时使用 dense 和 sparse encoder。"""
    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        sparse_encoder=mock_sparse_encoder,
        batch_size=2,
    )

    dense_vectors, sparse_vectors = processor.process(sample_chunks)

    # 两个 encoder 都应被调用
    assert mock_dense_encoder.encode.called
    assert mock_sparse_encoder.encode.called

    # 应返回正确数量的向量
    assert len(dense_vectors) == 5
    assert len(sparse_vectors) == 5


@pytest.mark.unit
def test_batch_processor_with_trace_context(mock_dense_encoder, sample_chunks):
    """测试传递 trace 上下文。"""
    mock_trace = Mock()
    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        batch_size=2,
    )

    processor.process(sample_chunks, trace=mock_trace)

    # 验证 trace 被传递给 encoder
    call_kwargs = mock_dense_encoder.encode.call_args[1]
    assert call_kwargs.get("trace") == mock_trace

    # 验证 trace.record_stage 被调用
    assert mock_trace.record_stage.called


@pytest.mark.unit
def test_batch_processor_single_batch(mock_dense_encoder):
    """测试单批次处理（chunks 数量 < batch_size）。"""
    chunks = [
        Chunk(id="1", text="text1", metadata={}),
        Chunk(id="2", text="text2", metadata={}),
    ]

    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        batch_size=10,  # 大于 chunks 数量
    )

    dense_vectors, _ = processor.process(chunks)

    # 应该只调用一次
    assert mock_dense_encoder.encode.call_count == 1

    # 应返回正确数量的向量
    assert len(dense_vectors) == 2


@pytest.mark.unit
def test_batch_processor_exact_batches(mock_dense_encoder):
    """测试恰好整除的批次。"""
    chunks = [
        Chunk(id=f"chunk-{i}", text=f"text{i}", metadata={})
        for i in range(6)
    ]

    processor = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        batch_size=2,
    )

    dense_vectors, _ = processor.process(chunks)

    # 6个chunks，batch_size=2，应该分3批，每批2个
    assert mock_dense_encoder.encode.call_count == 3

    # 验证每批的大小都是2
    for call_args in mock_dense_encoder.encode.call_args_list:
        assert len(call_args[0][0]) == 2


@pytest.mark.unit
def test_batch_processor_properties(mock_dense_encoder, mock_sparse_encoder):
    """测试属性方法。"""
    # 只有 dense encoder
    processor1 = BatchProcessor(dense_encoder=mock_dense_encoder)
    assert processor1.has_dense_encoder
    assert not processor1.has_sparse_encoder

    # 只有 sparse encoder
    processor2 = BatchProcessor(sparse_encoder=mock_sparse_encoder)
    assert not processor2.has_dense_encoder
    assert processor2.has_sparse_encoder

    # 两者都有
    processor3 = BatchProcessor(
        dense_encoder=mock_dense_encoder,
        sparse_encoder=mock_sparse_encoder,
    )
    assert processor3.has_dense_encoder
    assert processor3.has_sparse_encoder

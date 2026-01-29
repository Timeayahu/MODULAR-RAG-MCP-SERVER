"""SparseEncoder 测试。

根据 DEV_SPEC C9 验收标准：
- 输出结构可用于 bm25_indexer
- 对空文本有明确行为
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.models import Chunk


@pytest.fixture
def sample_chunks():
    """创建示例 Chunks。"""
    return [
        Chunk(
            id="chunk-1",
            text="machine learning is a subset of artificial intelligence",
            metadata={"source": "doc1.pdf"},
        ),
        Chunk(
            id="chunk-2",
            text="deep learning uses neural networks for pattern recognition",
            metadata={"source": "doc1.pdf"},
        ),
        Chunk(
            id="chunk-3",
            text="natural language processing enables computers to understand human language",
            metadata={"source": "doc2.pdf"},
        ),
    ]


@pytest.mark.unit
def test_sparse_encoder_basic_fit_and_encode(sample_chunks):
    """测试基本的 fit 和 encode 流程。"""
    encoder = SparseEncoder()

    # Fit
    encoder.fit(sample_chunks)
    assert encoder.is_fitted
    assert encoder.vocab_size > 0

    # Encode
    sparse_vectors = encoder.encode(sample_chunks)

    # 验收标准：输出结构可用于 bm25_indexer
    assert len(sparse_vectors) == len(sample_chunks)
    assert all(isinstance(sv, dict) for sv in sparse_vectors)
    assert all(
        all(isinstance(k, str) and isinstance(v, float) for k, v in sv.items())
        for sv in sparse_vectors
    )


@pytest.mark.unit
def test_sparse_encoder_empty_chunks_fit():
    """测试 fit 空 chunks 列表。"""
    encoder = SparseEncoder()
    encoder.fit([])

    # 应标记为 fitted，但统计为空
    assert encoder.is_fitted
    assert encoder.vocab_size == 0


@pytest.mark.unit
def test_sparse_encoder_empty_chunks_encode_raises_error():
    """测试 encode 空 chunks 列表应抛出错误。"""
    encoder = SparseEncoder()
    encoder.fit([Chunk(id="1", text="test", metadata={})])

    with pytest.raises(ValueError, match="chunks 不能为空"):
        encoder.encode([])


@pytest.mark.unit
def test_sparse_encoder_handles_empty_text():
    """验收标准：对空文本有明确行为。"""
    encoder = SparseEncoder()

    chunks = [
        Chunk(id="chunk-1", text="normal text here", metadata={}),
        Chunk(id="chunk-2", text="", metadata={}),  # 空文本
        Chunk(id="chunk-3", text="   ", metadata={}),  # 仅空格
    ]

    encoder.fit(chunks)
    sparse_vectors = encoder.encode(chunks)

    # 应返回 3 个向量
    assert len(sparse_vectors) == 3

    # 空文本应返回空字典
    assert sparse_vectors[1] == {}
    assert sparse_vectors[2] == {}

    # 正常文本应有权重
    assert len(sparse_vectors[0]) > 0


@pytest.mark.unit
def test_sparse_encoder_single_chunk():
    """测试编码单个 chunk。"""
    encoder = SparseEncoder()

    chunk = Chunk(
        id="chunk-1",
        text="machine learning algorithms",
        metadata={},
    )

    encoder.fit([chunk])
    weights = encoder.encode_single(chunk)

    # 应返回字典
    assert isinstance(weights, dict)
    assert len(weights) > 0
    assert "machine" in weights
    assert "learning" in weights
    assert "algorithms" in weights


@pytest.mark.unit
def test_sparse_encoder_stop_words_filtered():
    """测试停用词被过滤。"""
    encoder = SparseEncoder()

    chunk = Chunk(
        id="chunk-1",
        text="the machine learning is a powerful tool",
        metadata={},
    )

    encoder.fit([chunk])
    weights = encoder.encode_single(chunk)

    # 停用词不应出现在权重中
    assert "the" not in weights
    assert "is" not in weights
    assert "a" not in weights

    # 有意义的词应该存在
    assert "machine" in weights
    assert "learning" in weights
    assert "powerful" in weights
    assert "tool" in weights


@pytest.mark.unit
def test_sparse_encoder_custom_stop_words():
    """测试自定义停用词。"""
    custom_stop_words = {"machine", "learning"}
    encoder = SparseEncoder(stop_words=custom_stop_words)

    chunk = Chunk(
        id="chunk-1",
        text="machine learning is powerful",
        metadata={},
    )

    encoder.fit([chunk])
    weights = encoder.encode_single(chunk)

    # 自定义停用词不应出现
    assert "machine" not in weights
    assert "learning" not in weights

    # 其他词应该存在（默认停用词之外）
    assert "powerful" in weights


@pytest.mark.unit
def test_sparse_encoder_bm25_weights_reasonable(sample_chunks):
    """测试 BM25 权重的合理性。"""
    encoder = SparseEncoder()
    encoder.fit(sample_chunks)

    sparse_vectors = encoder.encode(sample_chunks)

    # 权重应该是正数
    for sv in sparse_vectors:
        for term, weight in sv.items():
            assert weight > 0, f"Term '{term}' has non-positive weight: {weight}"


@pytest.mark.unit
def test_sparse_encoder_idf_affects_weights(sample_chunks):
    """测试 IDF 影响权重（常见词权重应较低）。"""
    # 添加一个在所有文档中都出现的词
    chunks_with_common_term = [
        Chunk(id="1", text="common word appears here", metadata={}),
        Chunk(id="2", text="common word appears here too", metadata={}),
        Chunk(id="3", text="common word appears everywhere", metadata={}),
        Chunk(id="4", text="rare unique term", metadata={}),
    ]

    encoder = SparseEncoder()
    encoder.fit(chunks_with_common_term)

    # 编码包含常见词和稀有词的文档
    test_chunk = Chunk(
        id="test",
        text="common word and unique term",
        metadata={},
    )

    weights = encoder.encode_single(test_chunk)

    # 稀有词应该有更高的权重
    assert "unique" in weights
    assert "common" in weights
    assert weights["unique"] > weights["common"]


@pytest.mark.unit
def test_sparse_encoder_auto_fit_on_encode():
    """测试未 fit 时自动 fit。"""
    encoder = SparseEncoder()

    chunks = [
        Chunk(id="chunk-1", text="test content", metadata={}),
    ]

    # 直接 encode（未 fit）
    sparse_vectors = encoder.encode(chunks)

    # 应自动 fit 并编码
    assert encoder.is_fitted
    assert len(sparse_vectors) == 1


@pytest.mark.unit
def test_sparse_encoder_tokenization():
    """测试分词处理特殊字符和大小写。"""
    encoder = SparseEncoder()

    chunk = Chunk(
        id="chunk-1",
        text="Machine-Learning, Deep_Learning! NLP.",
        metadata={},
    )

    encoder.fit([chunk])
    weights = encoder.encode_single(chunk)

    # 应正确处理大小写和标点
    assert "machine" in weights
    assert "learning" in weights
    assert "deep" in weights
    assert "nlp" in weights


@pytest.mark.unit
def test_sparse_encoder_output_structure_for_indexer():
    """验收标准：确保输出结构可用于 bm25_indexer。

    BM25Indexer 需要的格式：
    - List[Dict[str, float]]
    - 每个 dict 是 {term: weight} 映射
    - term 是 str，weight 是 float
    """
    encoder = SparseEncoder()

    chunks = [
        Chunk(id="1", text="test document one", metadata={}),
        Chunk(id="2", text="test document two", metadata={}),
    ]

    encoder.fit(chunks)
    sparse_vectors = encoder.encode(chunks)

    # 验证输出结构
    assert isinstance(sparse_vectors, list)
    assert len(sparse_vectors) == 2

    for sv in sparse_vectors:
        assert isinstance(sv, dict)
        for term, weight in sv.items():
            assert isinstance(term, str)
            assert isinstance(weight, float)
            assert weight > 0  # BM25 权重应为正数

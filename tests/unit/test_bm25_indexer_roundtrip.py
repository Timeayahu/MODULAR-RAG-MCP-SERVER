"""BM25Indexer 测试 - 重点测试 roundtrip（构建-保存-加载-查询）。

根据 DEV_SPEC C12 验收标准：
- build 后能 load 并对同一语料查询返回稳定 top ids
"""

import shutil
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.storage.bm25_indexer import BM25Indexer


@pytest.fixture
def temp_index_dir(tmp_path):
    """创建临时索引目录。"""
    index_dir = tmp_path / "bm25_test"
    index_dir.mkdir(parents=True, exist_ok=True)
    yield str(index_dir)
    # 清理
    if index_dir.exists():
        shutil.rmtree(index_dir)


@pytest.fixture
def sample_data():
    """创建示例数据。"""
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
    sparse_vectors = [
        {"machine": 2.5, "learning": 2.0, "algorithm": 1.5},
        {"deep": 3.0, "learning": 2.5, "neural": 2.0},
        {"natural": 2.8, "language": 2.3, "processing": 2.0},
    ]
    metadata = [
        {"source": "doc1.pdf", "page": 1},
        {"source": "doc1.pdf", "page": 2},
        {"source": "doc2.pdf", "page": 1},
    ]
    return chunk_ids, sparse_vectors, metadata


@pytest.mark.unit
def test_bm25_indexer_basic_build(temp_index_dir, sample_data):
    """测试基本的索引构建。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)

    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 验证索引统计
    assert indexer.vocab_size > 0
    assert indexer.num_chunks == 3


@pytest.mark.unit
def test_bm25_indexer_query(temp_index_dir, sample_data):
    """测试查询功能。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 查询 "learning"
    query_terms = {"learning": 1.0}
    results = indexer.query(query_terms, top_k=2)

    # 应返回包含 "learning" 的 chunks
    assert len(results) == 2
    result_ids = [chunk_id for chunk_id, score in results]
    assert "chunk-1" in result_ids
    assert "chunk-2" in result_ids

    # 验证分数降序
    assert results[0][1] >= results[1][1]


@pytest.mark.unit
def test_bm25_indexer_roundtrip(temp_index_dir, sample_data):
    """验收标准：build 后能 load 并对同一语料查询返回稳定 top ids。"""
    chunk_ids, sparse_vectors, metadata = sample_data

    # 第一步：build 并 save
    indexer1 = BM25Indexer(temp_index_dir)
    indexer1.build(chunk_ids, sparse_vectors, metadata)
    indexer1.save("test_collection")

    # 第二步：创建新实例并 load
    indexer2 = BM25Indexer(temp_index_dir)
    indexer2.load("test_collection")

    # 验证索引统计一致
    assert indexer2.vocab_size == indexer1.vocab_size
    assert indexer2.num_chunks == indexer1.num_chunks

    # 第三步：查询并验证结果稳定
    query_terms = {"learning": 1.0, "machine": 0.8}

    results1 = indexer1.query(query_terms, top_k=3)
    results2 = indexer2.query(query_terms, top_k=3)

    # 验收标准：查询返回稳定的 top ids
    assert len(results1) == len(results2)
    for (id1, score1), (id2, score2) in zip(results1, results2):
        assert id1 == id2
        assert abs(score1 - score2) < 1e-6  # 分数也应该一致


@pytest.mark.unit
def test_bm25_indexer_empty_query(temp_index_dir, sample_data):
    """测试空查询。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 空查询应返回空列表
    results = indexer.query({})
    assert results == []


@pytest.mark.unit
def test_bm25_indexer_nonexistent_term(temp_index_dir, sample_data):
    """测试查询不存在的 term。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 查询不存在的 term
    query_terms = {"nonexistent": 1.0}
    results = indexer.query(query_terms)

    # 应返回空列表
    assert results == []


@pytest.mark.unit
def test_bm25_indexer_top_k_limit(temp_index_dir, sample_data):
    """测试 top_k 限制。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 查询一个常见 term
    query_terms = {"learning": 1.0}

    # 限制 top_k=1
    results = indexer.query(query_terms, top_k=1)
    assert len(results) == 1

    # 限制 top_k=10（但只有 2 个结果）
    results = indexer.query(query_terms, top_k=10)
    assert len(results) == 2  # 只返回实际匹配的数量


@pytest.mark.unit
def test_bm25_indexer_length_mismatch_raises_error(temp_index_dir):
    """长度不匹配应抛出错误。"""
    indexer = BM25Indexer(temp_index_dir)

    chunk_ids = ["chunk-1", "chunk-2"]
    sparse_vectors = [{"term": 1.0}]  # 长度不匹配

    with pytest.raises(ValueError, match="sparse_vectors 数量.*不一致"):
        indexer.build(chunk_ids, sparse_vectors)


@pytest.mark.unit
def test_bm25_indexer_save_and_load_files_exist(temp_index_dir, sample_data):
    """验证保存后文件存在。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    collection_name = "test_collection"
    indexer.save(collection_name)

    # 验证文件存在
    index_dir = Path(temp_index_dir)
    assert (index_dir / f"{collection_name}_index.json").exists()
    assert (index_dir / f"{collection_name}_chunks.json").exists()


@pytest.mark.unit
def test_bm25_indexer_load_nonexistent_raises_error(temp_index_dir):
    """加载不存在的索引应抛出错误。"""
    indexer = BM25Indexer(temp_index_dir)

    with pytest.raises(FileNotFoundError, match="索引文件不存在"):
        indexer.load("nonexistent_collection")


@pytest.mark.unit
def test_bm25_indexer_clear(temp_index_dir, sample_data):
    """测试清空索引。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    assert indexer.vocab_size > 0
    assert indexer.num_chunks > 0

    indexer.clear()

    assert indexer.vocab_size == 0
    assert indexer.num_chunks == 0


@pytest.mark.unit
def test_bm25_indexer_get_chunk_info(temp_index_dir, sample_data):
    """测试获取 chunk 信息。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 获取存在的 chunk 信息
    info = indexer.get_chunk_info("chunk-1")
    assert info is not None
    assert info["metadata"]["source"] == "doc1.pdf"

    # 获取不存在的 chunk 信息
    info = indexer.get_chunk_info("nonexistent")
    assert info is None


@pytest.mark.unit
def test_bm25_indexer_multi_term_query(temp_index_dir, sample_data):
    """测试多词查询。"""
    chunk_ids, sparse_vectors, metadata = sample_data
    indexer = BM25Indexer(temp_index_dir)
    indexer.build(chunk_ids, sparse_vectors, metadata)

    # 多词查询
    query_terms = {
        "learning": 1.0,
        "machine": 0.8,
        "deep": 0.6,
    }

    results = indexer.query(query_terms, top_k=3)

    # 应返回匹配的 chunks，按分数排序
    assert len(results) > 0
    assert all(isinstance(chunk_id, str) for chunk_id, score in results)
    assert all(isinstance(score, float) for chunk_id, score in results)

    # 验证分数降序
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_bm25_indexer_score_calculation(temp_index_dir):
    """测试分数计算逻辑。"""
    indexer = BM25Indexer(temp_index_dir)

    # 简单的测试数据
    chunk_ids = ["chunk-1", "chunk-2"]
    sparse_vectors = [
        {"term1": 2.0, "term2": 1.0},
        {"term1": 1.0, "term3": 3.0},
    ]

    indexer.build(chunk_ids, sparse_vectors)

    # 查询 term1
    query_terms = {"term1": 1.0}
    results = indexer.query(query_terms)

    # chunk-1 应该排在前面（term1 权重更高）
    assert results[0][0] == "chunk-1"
    assert results[0][1] == 2.0  # 1.0 * 2.0
    assert results[1][0] == "chunk-2"
    assert results[1][1] == 1.0  # 1.0 * 1.0

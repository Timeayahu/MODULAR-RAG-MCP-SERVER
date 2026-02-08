"""DenseRetriever 单元测试。

测试稠密向量检索功能，包括 VectorStore 调用和结果规范化。

根据 DEV_SPEC Task D2：
- 使用 mock VectorStore 和 Embedding
- 验证 score 规范化逻辑
- 验证 TraceContext 集成

pytest -q tests/unit/test_dense_retriever.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# 添加 src 到路径
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import pytest
from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.models import RetrievalCandidate


class TestDenseRetriever:
    """DenseRetriever 测试套件。"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """创建 mock VectorStore。"""
        mock_store = Mock()
        mock_store.provider_name = "mock_chroma"
        return mock_store
    
    @pytest.fixture
    def mock_embedding(self):
        """创建 mock Embedding。"""
        mock_emb = Mock()
        mock_emb.__class__.__name__ = "MockEmbedding"
        return mock_emb
    
    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding):
        """创建 DenseRetriever 实例。"""
        return DenseRetriever(
            vector_store=mock_vector_store,
            embedding=mock_embedding,
        )
    
    def test_init(self, retriever, mock_vector_store, mock_embedding):
        """测试：正确初始化。"""
        assert retriever.vector_store == mock_vector_store
        assert retriever.embedding == mock_embedding
    
    def test_retrieve_basic(self, retriever, mock_vector_store, mock_embedding):
        """测试：基本检索流程。"""
        # 配置 mock
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.query.return_value = [
            {
                "chunk_id": "chunk_1",
                "text": "Test content 1",
                "score": 0.95,
                "metadata": {"source": "doc1.pdf"},
            },
            {
                "chunk_id": "chunk_2",
                "text": "Test content 2",
                "score": 0.85,
                "metadata": {"source": "doc2.pdf"},
            },
        ]
        
        # 执行检索
        results = retriever.retrieve("test query", top_k=2)
        
        # 验证结果
        assert len(results) == 2
        assert all(isinstance(r, RetrievalCandidate) for r in results)
        
        # 验证第一个结果
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == "Test content 1"
        assert results[0].source == "dense"
        assert 0 <= results[0].score <= 1
        
        # 验证 embedding 被调用
        mock_embedding.embed.assert_called_once()
        assert mock_embedding.embed.call_args[0][0] == ["test query"]
        
        # 验证 vector store 被调用
        mock_vector_store.query.assert_called_once()
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["top_k"] == 2
    
    def test_retrieve_with_filters(self, retriever, mock_vector_store, mock_embedding):
        """测试：带过滤条件的检索。"""
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.query.return_value = []
        
        filters = {"source": "doc1.pdf", "year": 2024}
        retriever.retrieve("test query", top_k=5, filters=filters)
        
        # 验证 filters 被传递
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["filters"] == filters
    
    def test_retrieve_with_trace(self, retriever, mock_vector_store, mock_embedding):
        """测试：带 TraceContext 的检索。"""
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.query.return_value = [
            {"chunk_id": "c1", "text": "text", "score": 0.9, "metadata": {}},
        ]
        
        # 创建 mock TraceContext
        mock_trace = Mock()
        mock_trace.record_stage = Mock()
        
        retriever.retrieve("test query", top_k=1, trace=mock_trace)
        
        # 验证 trace 被调用
        mock_trace.record_stage.assert_called_once()
        call_args = mock_trace.record_stage.call_args[1]
        assert call_args["stage_name"] == "dense_retrieval"
        assert "num_results" in call_args["metrics"]
        assert "elapsed_time_ms" in call_args["metrics"]
    
    def test_retrieve_empty_query_raises_error(self, retriever):
        """测试：空查询抛出异常。"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("   ")
    
    def test_retrieve_embedding_failure(self, retriever, mock_embedding):
        """测试：Embedding 失败时抛出异常。"""
        mock_embedding.embed.side_effect = Exception("Embedding service unavailable")
        
        with pytest.raises(RuntimeError, match="Dense retrieval error"):
            retriever.retrieve("test query")
    
    def test_retrieve_vector_store_failure(self, retriever, mock_vector_store, mock_embedding):
        """测试：VectorStore 失败时抛出异常。"""
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.query.side_effect = Exception("Vector store connection error")
        
        with pytest.raises(RuntimeError, match="Dense retrieval error"):
            retriever.retrieve("test query")
    
    def test_normalize_score_cosine_similarity(self, retriever):
        """测试：余弦相似度分数规范化。"""
        # 测试范围内的值
        assert retriever._normalize_score(1.0) == 1.0  # 最相似
        assert retriever._normalize_score(0.0) == 0.5  # 中性
        assert retriever._normalize_score(-1.0) == 0.0  # 最不相似
        
        # 测试中间值
        assert 0.5 < retriever._normalize_score(0.5) < 1.0
        assert 0.0 < retriever._normalize_score(-0.5) < 0.5
    
    def test_normalize_score_out_of_range(self, retriever):
        """测试：超出范围的分数被裁剪。"""
        # 超出上界
        assert retriever._normalize_score(2.0) == 1.0
        assert retriever._normalize_score(10.0) == 1.0
        
        # 超出下界
        assert retriever._normalize_score(-2.0) == 0.0
        assert retriever._normalize_score(-10.0) == 0.0
    
    def test_normalize_results_handles_missing_fields(self, retriever):
        """测试：处理缺失字段的结果。"""
        raw_results = [
            {
                "id": "chunk_1",  # 使用 id 而非 chunk_id
                "text": "content",
                "score": 0.8,
            },
            {
                "chunk_id": "chunk_2",
                # 缺少 text 字段
                "score": 0.7,
                "metadata": {"key": "value"},
            },
        ]
        
        candidates = retriever._normalize_results(raw_results)
        
        assert len(candidates) == 2
        
        # 第一个结果使用 id 作为 chunk_id
        assert candidates[0].chunk_id == "chunk_1"
        assert candidates[0].text == "content"
        
        # 第二个结果缺少 text，应该有默认值
        assert candidates[1].chunk_id == "chunk_2"
        assert candidates[1].text == ""
        assert candidates[1].metadata == {"key": "value"}
    
    def test_retrieve_returns_empty_list_for_no_results(self, retriever, mock_vector_store, mock_embedding):
        """测试：无结果时返回空列表。"""
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.query.return_value = []
        
        results = retriever.retrieve("test query", top_k=10)
        
        assert results == []
        assert isinstance(results, list)


class TestDenseRetrieverIntegration:
    """DenseRetriever 集成测试（与真实 Trace 交互）。"""
    
    def test_retrieve_records_trace_metrics(self):
        """测试：正确记录 trace metrics。"""
        # 创建真实的 TraceContext mock（更接近实际使用）
        mock_trace = MagicMock()
        metrics_recorded = {}
        
        def capture_metrics(stage_name, metrics):
            metrics_recorded[stage_name] = metrics
        
        mock_trace.record_stage.side_effect = capture_metrics
        
        # 创建 retriever
        mock_store = Mock()
        mock_store.provider_name = "test"
        mock_store.query.return_value = [
            {"chunk_id": "c1", "text": "t1", "score": 0.9, "metadata": {}},
            {"chunk_id": "c2", "text": "t2", "score": 0.8, "metadata": {}},
        ]
        
        mock_emb = Mock()
        mock_emb.__class__.__name__ = "MockEmb"
        mock_emb.embed.return_value = [[0.1, 0.2, 0.3]]
        
        retriever = DenseRetriever(mock_store, mock_emb)
        
        # 执行检索
        retriever.retrieve("test query", top_k=5, filters={"year": 2024}, trace=mock_trace)
        
        # 验证记录的 metrics
        assert "dense_retrieval" in metrics_recorded
        metrics = metrics_recorded["dense_retrieval"]
        
        assert metrics["num_results"] == 2
        assert metrics["top_k"] == 5
        assert metrics["has_filters"] is True
        assert "elapsed_time_ms" in metrics
        assert metrics["elapsed_time_ms"] >= 0  # 时间可能很短，允许为 0

"""SparseRetriever 单元测试。

测试稀疏向量检索功能（BM25），包括索引加载和关键词查询。

根据 DEV_SPEC Task D3：
- 使用 mock BM25Indexer
- 验证关键词检索逻辑
- 验证 TraceContext 集成

pytest -p no:logfire tests/unit/test_sparse_retriever.py
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# 添加 src 到路径
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import pytest
from core.query_engine.sparse_retriever import SparseRetriever
from core.query_engine.models import RetrievalCandidate


class TestSparseRetriever:
    """SparseRetriever 测试套件。"""
    
    @pytest.fixture
    def mock_indexer(self):
        """创建 mock BM25Indexer。"""
        mock = Mock()
        mock.num_chunks = 100
        mock.vocab_size = 500
        return mock
    
    @pytest.fixture
    def retriever_no_auto_load(self, mock_indexer):
        """创建不自动加载的 SparseRetriever 实例。"""
        with patch('core.query_engine.sparse_retriever.BM25Indexer', return_value=mock_indexer):
            retriever = SparseRetriever(
                index_dir="data/db/bm25",
                collection_name="test",
                auto_load=False,
            )
            retriever.indexer = mock_indexer
            retriever._loaded = True  # 手动标记为已加载
            return retriever
    
    def test_init_no_auto_load(self):
        """测试：不自动加载的初始化。"""
        with patch('core.query_engine.sparse_retriever.BM25Indexer') as mock_cls:
            mock_indexer = Mock()
            mock_cls.return_value = mock_indexer
            
            retriever = SparseRetriever(
                index_dir="data/db/bm25",
                collection_name="test",
                auto_load=False,
            )
            
            assert retriever.collection_name == "test"
            assert retriever.is_loaded is False
            mock_indexer.load.assert_not_called()
    
    def test_retrieve_basic(self, retriever_no_auto_load, mock_indexer):
        """测试：基本检索流程。"""
        # 配置 mock
        mock_indexer.query.return_value = [
            ("chunk_1", 10.5),
            ("chunk_2", 8.3),
            ("chunk_3", 6.7),
        ]
        mock_indexer.get_chunk_info.side_effect = [
            {"metadata": {"source": "doc1.pdf", "page": 1}},
            {"metadata": {"source": "doc2.pdf", "page": 2}},
            {"metadata": {"source": "doc3.pdf", "page": 3}},
        ]
        
        # 执行检索
        results = retriever_no_auto_load.retrieve(
            keywords=["machine", "learning"],
            top_k=3,
        )
        
        # 验证结果
        assert len(results) == 3
        assert all(isinstance(r, RetrievalCandidate) for r in results)
        
        # 验证第一个结果
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 10.5
        assert results[0].source == "sparse"
        assert results[0].metadata == {"source": "doc1.pdf", "page": 1}
        
        # 验证 indexer.query 被正确调用
        mock_indexer.query.assert_called_once()
        call_kwargs = mock_indexer.query.call_args[1]
        assert call_kwargs["query_terms"] == {"machine": 1.0, "learning": 1.0}
        assert call_kwargs["top_k"] == 3
    
    def test_retrieve_with_trace(self, retriever_no_auto_load, mock_indexer):
        """测试：带 TraceContext 的检索。"""
        mock_indexer.query.return_value = [("chunk_1", 5.0)]
        mock_indexer.get_chunk_info.return_value = {"metadata": {}}
        
        # 创建 mock TraceContext
        mock_trace = Mock()
        mock_trace.record_stage = Mock()
        
        retriever_no_auto_load.retrieve(
            keywords=["test"],
            top_k=1,
            trace=mock_trace,
        )
        
        # 验证 trace 被调用
        mock_trace.record_stage.assert_called_once()
        call_args = mock_trace.record_stage.call_args[1]
        assert call_args["stage_name"] == "sparse_retrieval"
        assert "num_keywords" in call_args["metrics"]
        assert "elapsed_time_ms" in call_args["metrics"]
        assert call_args["metrics"]["num_keywords"] == 1
    
    def test_retrieve_empty_keywords_raises_error(self, retriever_no_auto_load):
        """测试：空关键词列表抛出异常。"""
        with pytest.raises(ValueError, match="Keywords cannot be empty"):
            retriever_no_auto_load.retrieve([])
    
    def test_retrieve_index_not_loaded_raises_error(self):
        """测试：索引未加载时抛出异常。"""
        with patch('core.query_engine.sparse_retriever.BM25Indexer') as mock_cls:
            mock_indexer = Mock()
            mock_indexer.load.side_effect = FileNotFoundError("Index not found")
            mock_cls.return_value = mock_indexer
            
            retriever = SparseRetriever(
                index_dir="data/db/bm25",
                collection_name="nonexistent",
                auto_load=False,
            )
            
            with pytest.raises(RuntimeError, match="BM25 index not found"):
                retriever.retrieve(keywords=["test"])
    
    def test_retrieve_query_failure(self, retriever_no_auto_load, mock_indexer):
        """测试：查询失败时抛出异常。"""
        mock_indexer.query.side_effect = Exception("Query failed")
        
        with pytest.raises(RuntimeError, match="Sparse retrieval error"):
            retriever_no_auto_load.retrieve(keywords=["test"])
    
    def test_retrieve_normalizes_keywords_to_lowercase(self, retriever_no_auto_load, mock_indexer):
        """测试：关键词被转换为小写。"""
        mock_indexer.query.return_value = []
        
        retriever_no_auto_load.retrieve(keywords=["Machine", "LEARNING"])
        
        # 验证查询 terms 是小写
        call_kwargs = mock_indexer.query.call_args[1]
        assert "machine" in call_kwargs["query_terms"]
        assert "learning" in call_kwargs["query_terms"]
        assert "Machine" not in call_kwargs["query_terms"]
    
    def test_retrieve_deduplicates_keywords(self, retriever_no_auto_load, mock_indexer):
        """测试：重复关键词被去重。"""
        mock_indexer.query.return_value = []
        
        retriever_no_auto_load.retrieve(keywords=["test", "Test", "TEST"])
        
        # 验证 query_terms 只有一个 "test"
        call_kwargs = mock_indexer.query.call_args[1]
        assert len(call_kwargs["query_terms"]) == 1
        assert "test" in call_kwargs["query_terms"]
    
    def test_retrieve_handles_missing_chunk_info(self, retriever_no_auto_load, mock_indexer):
        """测试：处理缺失的 chunk 信息。"""
        mock_indexer.query.return_value = [("chunk_1", 5.0)]
        mock_indexer.get_chunk_info.return_value = None
        
        results = retriever_no_auto_load.retrieve(keywords=["test"])
        
        assert len(results) == 1
        assert results[0].metadata == {}
    
    def test_retrieve_returns_empty_list_for_no_results(self, retriever_no_auto_load, mock_indexer):
        """测试：无结果时返回空列表。"""
        mock_indexer.query.return_value = []
        
        results = retriever_no_auto_load.retrieve(keywords=["nonexistent"])
        
        assert results == []
        assert isinstance(results, list)
    
    def test_reload_index(self, retriever_no_auto_load, mock_indexer):
        """测试：重新加载索引。"""
        mock_indexer.load.return_value = None
        mock_indexer.num_chunks = 200
        mock_indexer.vocab_size = 1000
        
        retriever_no_auto_load.reload_index(collection_name="new_collection")
        
        assert retriever_no_auto_load.collection_name == "new_collection"
        mock_indexer.load.assert_called_with(collection_name="new_collection")
        assert retriever_no_auto_load.is_loaded is True
    
    def test_properties(self, retriever_no_auto_load, mock_indexer):
        """测试：属性访问。"""
        assert retriever_no_auto_load.is_loaded is True
        assert retriever_no_auto_load.num_chunks == 100
        assert retriever_no_auto_load.vocab_size == 500
    
    def test_properties_when_not_loaded(self):
        """测试：未加载时的属性访问。"""
        with patch('core.query_engine.sparse_retriever.BM25Indexer') as mock_cls:
            mock_indexer = Mock()
            mock_cls.return_value = mock_indexer
            
            retriever = SparseRetriever(
                index_dir="data/db/bm25",
                collection_name="test",
                auto_load=False,
            )
            
            assert retriever.is_loaded is False
            assert retriever.num_chunks == 0
            assert retriever.vocab_size == 0


class TestSparseRetrieverIntegration:
    """SparseRetriever 集成测试（与 BM25Indexer 交互）。"""
    
    def test_auto_load_existing_index(self, tmp_path):
        """测试：自动加载已存在的索引。"""
        # 创建临时索引文件
        index_dir = tmp_path / "bm25"
        index_dir.mkdir()
        
        index_file = index_dir / "test_index.json"
        chunk_file = index_dir / "test_chunks.json"
        
        index_file.write_text('{"test": {"chunk_1": 1.0}}')
        chunk_file.write_text('{"chunk_1": {"index": 0, "metadata": {}}}')
        
        # 创建 retriever（应该自动加载）
        retriever = SparseRetriever(
            index_dir=str(index_dir),
            collection_name="test",
            auto_load=True,
        )
        
        assert retriever.is_loaded is True
        assert retriever.num_chunks == 1
        assert retriever.vocab_size == 1
    
    def test_auto_load_missing_index(self, tmp_path):
        """测试：索引不存在时不会失败。"""
        index_dir = tmp_path / "bm25_empty"
        index_dir.mkdir()
        
        # 创建 retriever（索引不存在，应该正常初始化）
        retriever = SparseRetriever(
            index_dir=str(index_dir),
            collection_name="nonexistent",
            auto_load=True,
        )
        
        assert retriever.is_loaded is False
        
        # 尝试查询应该抛出错误
        with pytest.raises(RuntimeError, match="BM25 index not found"):
            retriever.retrieve(keywords=["test"])

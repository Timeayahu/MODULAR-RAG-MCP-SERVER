"""Sparse Retriever - 稀疏向量检索模块（BM25）。

负责通过关键词匹配进行精确检索。
根据 DEV_SPEC Task D3：
- 从 data/db/bm25/ 载入索引
- 使用 BM25 算法进行关键词检索
- 支持 TraceContext 可观测性
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.query_engine.models import RetrievalCandidate
from ingestion.storage.bm25_indexer import BM25Indexer

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class SparseRetriever:
    """稀疏向量检索器（BM25）。
    
    通过关键词匹配在 BM25 索引中检索相关 chunks。
    
    设计原则：
    - 单一职责：只负责 BM25 关键词检索
    - 对称设计：与 DenseRetriever 保持相似接口
    - 延迟加载：索引在首次查询时加载
    - 可观测：支持 TraceContext 记录检索过程
    
    Attributes:
        indexer: BM25 索引器实例。
        index_dir: 索引目录路径。
        collection_name: 集合名称。
    """
    
    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        collection_name: str = "default",
        auto_load: bool = True,
    ) -> None:
        """初始化 SparseRetriever。
        
        Args:
            index_dir: BM25 索引目录路径。
            collection_name: 集合名称（用于多集合支持）。
            auto_load: 是否在初始化时自动加载索引（默认 True）。
        """
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.indexer = BM25Indexer(index_dir=str(self.index_dir))
        self._loaded = False
        
        # 如果启用自动加载且索引文件存在，则加载
        if auto_load:
            index_file = self.index_dir / f"{collection_name}_index.json"
            if index_file.exists():
                try:
                    self.indexer.load(collection_name=collection_name)
                    self._loaded = True
                    logger.info(
                        f"SparseRetriever initialized and loaded: "
                        f"{self.indexer.num_chunks} chunks, {self.indexer.vocab_size} terms"
                    )
                except Exception as e:
                    logger.warning(f"Failed to auto-load index: {e}. Will retry on first query.")
            else:
                logger.info(
                    f"SparseRetriever initialized without index (file not found: {index_file})"
                )
        else:
            logger.info("SparseRetriever initialized (auto_load=False)")
    
    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        trace: Optional["TraceContext"] = None,
    ) -> List[RetrievalCandidate]:
        """执行稀疏向量检索（BM25）。
        
        流程：
        1. 确保索引已加载
        2. 将关键词转换为查询 terms（均等权重）
        3. 调用 BM25Indexer.query() 检索 Top-K
        4. 规范化分数并封装为 RetrievalCandidate
        
        Args:
            keywords: 关键词列表。
            top_k: 返回的候选数量（默认 10）。
            trace: 追踪上下文（可选）。
        
        Returns:
            检索候选列表，按 BM25 分数降序排列。
        
        Raises:
            ValueError: 如果 keywords 为空。
            RuntimeError: 如果索引未加载或查询失败。
        """
        if not keywords:
            raise ValueError("Keywords cannot be empty")
        
        # 确保索引已加载
        self._ensure_loaded()
        
        start_time = time.time()
        
        try:
            # Step 1: 构建查询 terms（均等权重）
            query_terms = {kw.lower(): 1.0 for kw in keywords}
            
            logger.debug(
                f"Querying BM25 index with {len(query_terms)} terms, top_k={top_k}"
            )
            
            # Step 2: 查询 BM25 索引
            raw_results = self.indexer.query(
                query_terms=query_terms,
                top_k=top_k,
            )
            
            # Step 3: 转换为 RetrievalCandidate
            candidates = self._convert_to_candidates(raw_results)
            
            elapsed_time = time.time() - start_time
            
            # 记录到 TraceContext
            if trace:
                trace.record_stage(
                    stage_name="sparse_retrieval",
                    metrics={
                        "num_keywords": len(keywords),
                        "num_query_terms": len(query_terms),
                        "top_k": top_k,
                        "num_results": len(candidates),
                        "elapsed_time_ms": round(elapsed_time * 1000, 2),
                    },
                )
            
            logger.info(
                f"Sparse retrieval completed: {len(candidates)} candidates in {elapsed_time:.3f}s"
            )
            
            return candidates
        
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"Sparse retrieval error: {e}") from e
    
    def _ensure_loaded(self) -> None:
        """确保索引已加载。
        
        Raises:
            RuntimeError: 如果索引文件不存在或加载失败。
        """
        if self._loaded:
            return
        
        try:
            self.indexer.load(collection_name=self.collection_name)
            self._loaded = True
            logger.info(
                f"Index loaded on-demand: {self.indexer.num_chunks} chunks, "
                f"{self.indexer.vocab_size} terms"
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"BM25 index not found for collection '{self.collection_name}'. "
                f"Please build the index first. Path: {self.index_dir}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load BM25 index: {e}") from e
    
    def _convert_to_candidates(
        self,
        raw_results: List[tuple],
    ) -> List[RetrievalCandidate]:
        """将 BM25Indexer 返回的原始结果转换为 RetrievalCandidate。
        
        Args:
            raw_results: BM25Indexer.query() 返回的 [(chunk_id, score)] 列表。
        
        Returns:
            规范化后的候选列表。
        """
        candidates = []
        
        for chunk_id, score in raw_results:
            # 获取 chunk 信息
            chunk_info = self.indexer.get_chunk_info(chunk_id)
            
            # BM25 分数可能很大，需要规范化
            # 这里使用简单的策略：保留原始分数（后续在 fusion 阶段会统一处理）
            normalized_score = float(score)
            
            candidate = RetrievalCandidate(
                chunk_id=chunk_id,
                text="",  # BM25Indexer 不存储文本，需要从其他地方获取
                score=normalized_score,
                metadata=chunk_info.get("metadata", {}) if chunk_info else {},
                source="sparse",
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def reload_index(self, collection_name: Optional[str] = None) -> None:
        """重新加载索引。
        
        Args:
            collection_name: 集合名称，如果为 None 则使用当前集合。
        
        Raises:
            RuntimeError: 如果加载失败。
        """
        if collection_name is not None:
            self.collection_name = collection_name
        
        try:
            self.indexer.load(collection_name=self.collection_name)
            self._loaded = True
            logger.info(
                f"Index reloaded: {self.indexer.num_chunks} chunks, "
                f"{self.indexer.vocab_size} terms (collection: {self.collection_name})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to reload index: {e}") from e
    
    @property
    def is_loaded(self) -> bool:
        """返回索引是否已加载。"""
        return self._loaded
    
    @property
    def num_chunks(self) -> int:
        """返回索引中的 chunk 数量。"""
        return self.indexer.num_chunks if self._loaded else 0
    
    @property
    def vocab_size(self) -> int:
        """返回索引的词汇表大小。"""
        return self.indexer.vocab_size if self._loaded else 0

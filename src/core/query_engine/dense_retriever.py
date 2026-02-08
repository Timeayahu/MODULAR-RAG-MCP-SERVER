"""Dense Retriever - 稠密向量检索模块。

负责通过 Embedding 向量进行语义相似度检索。
根据 DEV_SPEC Task D2：
- 调用 VectorStore.query() 进行向量检索
- 透传并规范化 score
- 支持 TraceContext 可观测性
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.query_engine.models import RetrievalCandidate
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class DenseRetriever:
    """稠密向量检索器。
    
    通过 query 的 embedding 向量在向量数据库中检索语义相似的 chunks。
    
    设计原则：
    - 单一职责：只负责向量相似度检索
    - 依赖注入：接收外部创建的 VectorStore 和 Embedding 实例
    - 透传与规范化：保持原始结果，规范化 score 到 [0, 1]
    - 可观测：支持 TraceContext 记录检索过程
    
    Attributes:
        vector_store: 向量数据库实例。
        embedding: Embedding 模型实例。
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding: BaseEmbedding,
    ) -> None:
        """初始化 DenseRetriever。
        
        Args:
            vector_store: 向量数据库实例（用于查询）。
            embedding: Embedding 模型实例（用于生成查询向量）。
        """
        self.vector_store = vector_store
        self.embedding = embedding
        logger.info(
            f"DenseRetriever initialized with vector_store={vector_store.provider_name}, "
            f"embedding={embedding.__class__.__name__}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional["TraceContext"] = None,
    ) -> List[RetrievalCandidate]:
        """执行稠密向量检索。
        
        流程：
        1. 使用 Embedding 模型将 query 转换为向量
        2. 调用 VectorStore.query() 检索 Top-K 相似 chunks
        3. 规范化 score 并封装为 RetrievalCandidate
        
        Args:
            query: 查询字符串。
            top_k: 返回的候选数量（默认 10）。
            filters: 元数据过滤条件（可选）。
            trace: 追踪上下文（可选）。
        
        Returns:
            检索候选列表，按相似度分数降序排列。
        
        Raises:
            ValueError: 如果 query 为空。
            RuntimeError: 如果 Embedding 或 VectorStore 调用失败。
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Step 1: 生成查询向量
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self._generate_query_embedding(query, trace)
            
            # Step 2: 向量检索
            logger.debug(f"Querying vector store with top_k={top_k}, filters={filters}")
            raw_results = self.vector_store.query(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters or {},
                trace=trace,
            )
            
            # Step 3: 规范化并转换为 RetrievalCandidate
            candidates = self._normalize_results(raw_results)
            
            elapsed_time = time.time() - start_time
            
            # 记录到 TraceContext
            if trace:
                trace.record_stage(
                    stage_name="dense_retrieval",
                    metrics={
                        "query_length": len(query),
                        "top_k": top_k,
                        "num_results": len(candidates),
                        "elapsed_time_ms": round(elapsed_time * 1000, 2),
                        "has_filters": bool(filters),
                    },
                )
            
            logger.info(
                f"Dense retrieval completed: {len(candidates)} candidates in {elapsed_time:.3f}s"
            )
            
            return candidates
        
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"Dense retrieval error: {e}") from e
    
    def _generate_query_embedding(
        self,
        query: str,
        trace: Optional["TraceContext"] = None,
    ) -> List[float]:
        """生成查询的 embedding 向量。
        
        Args:
            query: 查询字符串。
            trace: 追踪上下文（可选）。
        
        Returns:
            查询的向量表示。
        
        Raises:
            RuntimeError: 如果 Embedding 生成失败。
        """
        try:
            # embed() 方法返回 List[List[float]]（批量），我们只取第一个
            embeddings = self.embedding.embed([query], trace=trace)
            
            if not embeddings or not embeddings[0]:
                raise RuntimeError("Embedding returned empty result")
            
            return embeddings[0]
        
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    def _normalize_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[RetrievalCandidate]:
        """规范化 VectorStore 返回的原始结果。
        
        将 VectorStore 的返回格式转换为统一的 RetrievalCandidate 格式，
        并确保 score 在 [0, 1] 范围内。
        
        Args:
            raw_results: VectorStore 返回的原始结果列表。
        
        Returns:
            规范化后的候选列表。
        """
        candidates = []
        
        for result in raw_results:
            # 提取必需字段
            chunk_id = result.get("chunk_id") or result.get("id", "")
            text = result.get("text", "")
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            # 规范化 score 到 [0, 1]
            # 注意：不同向量库的分数范围可能不同
            # - 余弦相似度：[-1, 1] → 转换为 [0, 1]
            # - 欧氏距离：[0, +∞) → 需要转换（距离越小越相似）
            normalized_score = self._normalize_score(score)
            
            candidate = RetrievalCandidate(
                chunk_id=chunk_id,
                text=text,
                score=normalized_score,
                metadata=metadata,
                source="dense",
            )
            
            candidates.append(candidate)
        
        return candidates
    
    @staticmethod
    def _normalize_score(score: float) -> float:
        """规范化相似度分数到 [0, 1] 区间。
        
        不同向量库的 score 语义不同：
        - 余弦相似度：[-1, 1]，越大越相似
        - L2 距离：[0, +∞)，越小越相似
        
        这里假设输入是余弦相似度（Chroma 默认使用余弦）。
        
        Args:
            score: 原始分数。
        
        Returns:
            规范化到 [0, 1] 的分数。
        """
        # 假设是余弦相似度：[-1, 1] → [0, 1]
        # 转换公式：(score + 1) / 2
        if score < -1:
            score = -1.0
        elif score > 1:
            score = 1.0
        
        normalized = (score + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))  # 确保在 [0, 1] 范围内

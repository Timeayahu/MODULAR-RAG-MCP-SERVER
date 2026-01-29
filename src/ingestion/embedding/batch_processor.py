"""BatchProcessor 实现 - 批处理编排器。

根据 DEV_SPEC 3.1.1 Embedding 阶段：
- 批处理优化：所有计算均采用 batch_size 驱动的批处理模式，
  最大化 CPU 利用率并减少网络 RTT。

本模块负责：
1. 将 chunks 分批
2. 驱动 dense/sparse 编码
3. 记录批次耗时（为 trace 预留）
4. 保持 chunks 顺序
"""

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ingestion.models import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext
    from ingestion.embedding.dense_encoder import DenseEncoder
    from ingestion.embedding.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批处理编排器。

    特性：
    - 将 chunks 分批处理
    - 并行驱动 dense 和 sparse 编码
    - 记录批次耗时
    - 保持 chunks 顺序
    """

    def __init__(
        self,
        dense_encoder: Optional["DenseEncoder"] = None,
        sparse_encoder: Optional["SparseEncoder"] = None,
        batch_size: int = 32,
    ) -> None:
        """初始化 BatchProcessor。

        Args:
            dense_encoder: DenseEncoder 实例（可选）。
            sparse_encoder: SparseEncoder 实例（可选）。
            batch_size: 批次大小，默认 32。
        """
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前值: {batch_size}")

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """批处理 chunks，执行 dense 和 sparse 编码。

        Args:
            chunks: Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            (dense_vectors, sparse_vectors) 元组。

        Raises:
            ValueError: 如果 chunks 为空或两个 encoder 都未配置。
        """
        if not chunks:
            raise ValueError("chunks 不能为空")

        if self.dense_encoder is None and self.sparse_encoder is None:
            raise ValueError("至少需要配置一个 encoder")

        # 分批
        batches = self._create_batches(chunks)

        logger.info(
            f"开始批处理: {len(chunks)} chunks, "
            f"{len(batches)} batches (batch_size={self.batch_size})"
        )

        # 处理所有批次
        all_dense_vectors: List[List[float]] = []
        all_sparse_vectors: List[Dict[str, float]] = []

        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()

            # Dense 编码
            if self.dense_encoder is not None:
                dense_vectors = self.dense_encoder.encode(batch, trace=trace)
                all_dense_vectors.extend(dense_vectors)
            else:
                # 如果不需要 dense 编码，填充空向量
                all_dense_vectors.extend([[] for _ in batch])

            # Sparse 编码
            if self.sparse_encoder is not None:
                sparse_vectors = self.sparse_encoder.encode(batch, trace=trace)
                all_sparse_vectors.extend(sparse_vectors)
            else:
                # 如果不需要 sparse 编码，填充空字典
                all_sparse_vectors.extend([{} for _ in batch])

            batch_elapsed = time.time() - batch_start

            logger.debug(
                f"批次 {batch_idx + 1}/{len(batches)} 完成: "
                f"{len(batch)} chunks, {batch_elapsed:.3f}s"
            )

            # 记录到 trace（如果提供）
            if trace is not None:
                trace.record_stage(
                    stage_name="batch_encoding",
                    details={
                        "batch_index": batch_idx,
                        "batch_size": len(batch),
                        "elapsed": batch_elapsed,
                    },
                )

        # 验证输出长度
        if len(all_dense_vectors) != len(chunks):
            raise RuntimeError(
                f"Dense vectors 数量 ({len(all_dense_vectors)}) "
                f"与 chunks 数量 ({len(chunks)}) 不一致"
            )

        if len(all_sparse_vectors) != len(chunks):
            raise RuntimeError(
                f"Sparse vectors 数量 ({len(all_sparse_vectors)}) "
                f"与 chunks 数量 ({len(chunks)}) 不一致"
            )

        logger.info(f"批处理完成: {len(chunks)} chunks processed")

        return all_dense_vectors, all_sparse_vectors

    def process_dense_only(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> List[List[float]]:
        """仅执行 dense 编码。

        Args:
            chunks: Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            Dense vectors。
        """
        if self.dense_encoder is None:
            raise ValueError("dense_encoder 未配置")

        dense_vectors, _ = self.process(chunks, trace=trace)
        return dense_vectors

    def process_sparse_only(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, float]]:
        """仅执行 sparse 编码。

        Args:
            chunks: Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            Sparse vectors。
        """
        if self.sparse_encoder is None:
            raise ValueError("sparse_encoder 未配置")

        _, sparse_vectors = self.process(chunks, trace=trace)
        return sparse_vectors

    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """将 chunks 分批。

        Args:
            chunks: Chunk 列表。

        Returns:
            批次列表，每个批次是 chunk 列表。
        """
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batches.append(batch)
        return batches

    @property
    def has_dense_encoder(self) -> bool:
        """返回是否配置了 dense encoder。"""
        return self.dense_encoder is not None

    @property
    def has_sparse_encoder(self) -> bool:
        """返回是否配置了 sparse encoder。"""
        return self.sparse_encoder is not None

"""DenseEncoder 实现 - 稠密向量编码器。

根据 DEV_SPEC 3.1.1 Embedding 阶段：
- Dense Embeddings（语义向量）：调用 Embedding 模型（如 OpenAI text-embedding-3 或 BGE）
  生成高维浮点向量，捕捉文本的深层语义关联，解决"词不同意同"的检索难题。
- 批处理优化：所有计算均采用 batch_size 驱动的批处理模式，最大化 CPU 利用率并减少网络 RTT。
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from ingestion.models import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext
    from libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class DenseEncoder:
    """稠密向量编码器。

    特性：
    - 批量处理 chunks 文本
    - 调用 BaseEmbedding 进行向量化
    - 支持 trace 上下文传递
    - 处理空文本和异常情况
    """

    def __init__(self, embedding: "BaseEmbedding") -> None:
        """初始化 DenseEncoder。

        Args:
            embedding: BaseEmbedding 实例。
        """
        self.embedding = embedding

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> List[List[float]]:
        """将 Chunks 编码为稠密向量。

        Args:
            chunks: Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            向量列表，长度与 chunks 一致。

        Raises:
            ValueError: 如果 chunks 为空。
        """
        if not chunks:
            raise ValueError("chunks 不能为空")

        # 提取文本
        texts = [chunk.text for chunk in chunks]

        # 处理空文本
        if any(not text.strip() for text in texts):
            logger.warning("检测到空文本 chunk，将使用空字符串占位")

        # 批量向量化
        try:
            vectors = self.embedding.embed(texts, trace=trace)
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            raise

        # 验证输出
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"向量数量 ({len(vectors)}) 与 chunks 数量 ({len(chunks)}) 不一致"
            )

        return vectors

    def encode_single(
        self,
        chunk: Chunk,
        trace: Optional["TraceContext"] = None,
    ) -> List[float]:
        """编码单个 Chunk。

        Args:
            chunk: 单个 Chunk。
            trace: 可选追踪上下文。

        Returns:
            向量。
        """
        vectors = self.encode([chunk], trace=trace)
        return vectors[0]

    @property
    def embedding_dim(self) -> Optional[int]:
        """返回向量维度（如果可获取）。

        Returns:
            向量维度，如果无法获取则返回 None。
        """
        # 尝试通过 config 获取维度
        if hasattr(self.embedding, "config"):
            config = self.embedding.config
            if hasattr(config, "dimension"):
                return config.dimension
        return None

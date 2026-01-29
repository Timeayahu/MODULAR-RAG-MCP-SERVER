"""Embedding 抽象基类定义。

本模块基于 LlamaIndex 的 BaseEmbedding 抽象进行封装，
提供与 LlamaIndex 生态无缝集成的 Embedding 接口。

根据 DEV_SPEC 3.3.3：
- LlamaIndex 提供了 BaseEmbedding 抽象接口，支持不同 Embedding 模型的可插拔替换
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from core.settings import EmbeddingConfig

# LlamaIndex Embedding 类型导入
try:
    from llama_index.core.embeddings import BaseEmbedding as LlamaBaseEmbedding

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    LlamaBaseEmbedding = None  # type: ignore

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext
    from llama_index.core.embeddings import BaseEmbedding as LlamaBaseEmbedding


class BaseEmbedding(ABC):
    """Embedding 抽象基类，基于 LlamaIndex BaseEmbedding 接口设计。

    设计说明：
    - 提供与 LlamaIndex BaseEmbedding 兼容的接口
    - 支持通过 get_llama_embedding() 获取原生 LlamaIndex Embedding 实例
    - 保持简洁的 embed() 方法用于快速调用
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """初始化 Embedding。

        Args:
            config: Embedding 配置。
        """
        self._config = config
        self._llama_embedding: Optional["LlamaBaseEmbedding"] = None

    @property
    def config(self) -> EmbeddingConfig:
        """返回 Embedding 配置。"""
        return self._config

    @abstractmethod
    def get_llama_embedding(self) -> "LlamaBaseEmbedding":
        """获取 LlamaIndex 原生 Embedding 实例。

        Returns:
            LlamaIndex BaseEmbedding 实例。

        Raises:
            ImportError: 如果 llama-index 未安装。
        """
        raise NotImplementedError

    def embed(
        self,
        texts: List[str],
        trace: Optional["TraceContext"] = None,
    ) -> List[List[float]]:
        """批量向量化文本。

        Args:
            texts: 输入文本列表。
            trace: 可选追踪上下文。

        Returns:
            向量列表，长度与 texts 一致。
        """
        if not texts:
            return []

        embedding = self.get_llama_embedding()

        # 使用 LlamaIndex 的批量 embed 方法
        # LlamaIndex 的 get_text_embedding_batch 会自动处理批次
        vectors = embedding.get_text_embedding_batch(texts)

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """向量化单个查询文本。

        查询文本的向量化可能与文档文本不同（取决于模型）。

        Args:
            text: 查询文本。

        Returns:
            向量。
        """
        embedding = self.get_llama_embedding()
        return embedding.get_query_embedding(text)

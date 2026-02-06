"""VectorStore 抽象基类定义。

本模块基于 LlamaIndex 的 VectorStore 抽象进行封装，
提供与 LlamaIndex 生态无缝集成的向量数据库接口。

根据 DEV_SPEC 3.3.3：
- LlamaIndex 为向量数据库定义了统一的 VectorStore 抽象接口，
  所有主流向量库（Chroma、Qdrant、Pinecone 等）都有对应适配器
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.settings import VectorStoreConfig

# LlamaIndex VectorStore 类型导入
try:
    from llama_index.core.vector_stores import VectorStore as LlamaVectorStore
    from llama_index.core.schema import TextNode

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    LlamaVectorStore = None  # type: ignore
    TextNode = None  # type: ignore

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext
    from llama_index.core.vector_stores import VectorStore as LlamaVectorStore


class BaseVectorStore(ABC):
    """向量数据库抽象基类，基于 LlamaIndex VectorStore 接口设计。

    设计说明：
    - 提供与 LlamaIndex VectorStore 兼容的接口
    - 支持通过 get_llama_vector_store() 获取原生 LlamaIndex VectorStore 实例
    - 保持简洁的 upsert() 和 query() 方法用于快速调用

    根据 DEV_SPEC 3.1.1 存储策略：
    - All-in-One 存储：每条记录同时包含 Dense Vector、Sparse Vector、
      Chunk 原始文本和 Metadata
    - 幂等性设计：基于 chunk_id 的 upsert 语义
    """

    provider_name: str = "base"

    def __init__(self, config: VectorStoreConfig) -> None:
        """初始化 VectorStore。

        Args:
            config: VectorStore 配置。
        """
        self._config = config
        self._llama_vector_store: Optional["LlamaVectorStore"] = None

    @property
    def config(self) -> VectorStoreConfig:
        """返回 VectorStore 配置。"""
        return self._config

    @abstractmethod
    def get_llama_vector_store(self) -> "LlamaVectorStore":
        """获取 LlamaIndex 原生 VectorStore 实例。

        Returns:
            LlamaIndex VectorStore 实例。

        Raises:
            ImportError: 如果 llama-index 未安装。
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional["TraceContext"] = None,
    ) -> None:
        """写入或更新向量记录。

        Args:
            records: 记录列表，约定包含以下字段：
                - id: str，记录唯一标识
                - vector: List[float]，稠密向量
                - text: str，原始文本
                - metadata: Dict，元数据
            trace: 可选追踪上下文。

        Note:
            此方法同时支持直接调用和通过 LlamaIndex VectorStoreIndex 调用。
            当通过 LlamaIndex 调用时，会自动转换 TextNode 为 records 格式。
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, Any]]:
        """向量查询。

        Args:
            vector: 查询向量。
            top_k: 返回数量。
            filters: 可选过滤条件（对应 LlamaIndex 的 MetadataFilters）。
            trace: 可选追踪上下文。

        Returns:
            命中记录列表，约定包含以下字段：
                - id: str，记录标识
                - score: float，相似度分数
                - text: str，原始文本
                - metadata: Dict，元数据
        """
        raise NotImplementedError


    def delete(
        self,
        ids: List[str],
        trace: Optional["TraceContext"] = None,
    ) -> None:
        """删除指定记录。

        Args:
            ids: 要删除的记录 ID 列表。
            trace: 可选追踪上下文。
        """
        # 默认实现：子类可覆盖
        raise NotImplementedError(f"{self.provider_name} 不支持 delete 操作")

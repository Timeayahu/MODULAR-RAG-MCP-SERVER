"""VectorStore 工厂：根据配置创建对应的 VectorStore 实例。

根据 DEV_SPEC 3.3.3：
- LlamaIndex 为向量数据库定义了统一的 VectorStore 抽象接口
- 通过 VectorStoreFactory 根据配置选择具体实现

本工厂基于 LlamaIndex VectorStore 抽象层，支持以下 Provider：
- chroma: Chroma 嵌入式向量数据库（默认）

架构设计上预留了扩展能力，未来可支持：
- qdrant: Qdrant 向量数据库
- pinecone: Pinecone 云端向量数据库
"""

from typing import Dict, Type

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore


class VectorStoreFactory:
    """VectorStore 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseVectorStore]] = {"chroma": ChromaStore}

    @classmethod
    def register(cls, provider: str, store_cls: Type[BaseVectorStore]) -> None:
        """注册 VectorStore 实现。"""
        cls._registry[provider.lower()] = store_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """取消注册 VectorStore 实现。"""
        cls._registry.pop(provider.lower(), None)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（测试辅助）。"""
        cls._registry.clear()

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """根据配置创建 VectorStore 实例。

        Args:
            settings: 全局配置。

        Returns:
            BaseVectorStore 实例。

        Raises:
            ValueError: provider 未注册。
        """
        provider = settings.vector_store.provider.lower()
        if provider not in cls._registry:
            raise ValueError(f"未知的 VectorStore provider: {provider}")
        return cls._registry[provider](settings.vector_store)

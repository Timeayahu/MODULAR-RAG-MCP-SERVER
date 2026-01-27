"""Embedding 工厂：根据配置创建对应的 Embedding 实例。"""

from typing import Dict, Type

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.local_embedding import LocalEmbedding
from libs.embedding.openai_embedding import OpenAIEmbedding


class EmbeddingFactory:
    """Embedding 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseEmbedding]] = {
        "openai": OpenAIEmbedding,
        "local": LocalEmbedding,
    }

    @classmethod
    def register(cls, provider: str, embed_cls: Type[BaseEmbedding]) -> None:
        """注册 Embedding 实现。"""
        cls._registry[provider.lower()] = embed_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """取消注册 Embedding 实现。"""
        cls._registry.pop(provider.lower(), None)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（测试辅助）。"""
        cls._registry.clear()

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """根据配置创建 Embedding 实例。

        Args:
            settings: 全局配置。

        Returns:
            BaseEmbedding 实例。

        Raises:
            ValueError: provider 未注册。
        """
        provider = settings.embedding.provider.lower()
        if provider not in cls._registry:
            raise ValueError(f"未知的 Embedding provider: {provider}")
        return cls._registry[provider](settings.embedding)

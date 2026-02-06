"""Embedding 工厂：根据配置创建对应的 Embedding 实例。

根据 DEV_SPEC 3.3.3：
- LlamaIndex 提供了 BaseEmbedding 抽象接口，支持不同 Embedding 模型的可插拔替换
- 支持云端服务（OpenAI Embedding）与本地模型（Sentence-Transformers, BGE）自由切换

本工厂基于 LlamaIndex Embedding 抽象层，支持以下 Provider：
- openai: OpenAI Embedding API (text-embedding-3-small/large)
- local: 本地 HuggingFace 模型 (BGE, Sentence-Transformers)
- qwen: Qwen Embedding API (OpenAI-Compatible)
- fake: 测试用固定向量
"""

from typing import Dict, Type

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.local_embedding import LocalEmbedding, FakeEmbedding
from libs.embedding.openai_embedding import OpenAIEmbedding
from libs.embedding.qwen_embedding import QwenEmbedding


class EmbeddingFactory:
    """Embedding 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseEmbedding]] = {
        "openai": OpenAIEmbedding,
        "local": LocalEmbedding,
        "qwen": QwenEmbedding,
        "fake": FakeEmbedding,
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

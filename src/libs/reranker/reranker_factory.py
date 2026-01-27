"""Reranker 工厂：根据配置创建对应的 Reranker 实例。"""

from typing import Dict, Type

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.none_reranker import NoneReranker


class RerankerFactory:
    """Reranker 工厂，按 backend 路由到具体实现。"""

    _registry: Dict[str, Type[BaseReranker]] = {
        "none": NoneReranker,
        "cross_encoder": CrossEncoderReranker,
    }

    @classmethod
    def register(cls, backend: str, reranker_cls: Type[BaseReranker]) -> None:
        """注册 Reranker 实现。"""
        cls._registry[backend.lower()] = reranker_cls

    @classmethod
    def unregister(cls, backend: str) -> None:
        """取消注册 Reranker 实现。"""
        cls._registry.pop(backend.lower(), None)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（测试辅助）。"""
        cls._registry.clear()

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """根据配置创建 Reranker 实例。

        Args:
            settings: 全局配置。

        Returns:
            BaseReranker 实例。

        Raises:
            ValueError: backend 未注册。
        """
        backend = settings.rerank.backend.lower()
        if backend == "llm":
            return LLMReranker.from_settings(settings)
        if backend not in cls._registry:
            raise ValueError(f"未知的 Reranker backend: {backend}")
        return cls._registry[backend](settings.rerank)

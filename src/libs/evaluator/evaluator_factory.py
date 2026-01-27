"""Evaluator 工厂：根据配置创建对应的 Evaluator 实例。"""

from typing import Dict, Type

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """Evaluator 工厂，按 backend 路由到具体实现。"""

    _registry: Dict[str, Type[BaseEvaluator]] = {"custom": CustomEvaluator}

    @classmethod
    def register(cls, backend: str, evaluator_cls: Type[BaseEvaluator]) -> None:
        """注册 Evaluator 实现。"""
        cls._registry[backend.lower()] = evaluator_cls

    @classmethod
    def unregister(cls, backend: str) -> None:
        """取消注册 Evaluator 实现。"""
        cls._registry.pop(backend.lower(), None)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（测试辅助）。"""
        cls._registry.clear()

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """根据配置创建 Evaluator 实例。

        Args:
            settings: 全局配置。

        Returns:
            BaseEvaluator 实例。

        Raises:
            ValueError: backend 未注册。
        """
        backend = settings.evaluation.backend.lower()
        if backend not in cls._registry:
            raise ValueError(f"未知的 Evaluator backend: {backend}")
        return cls._registry[backend](settings.evaluation)

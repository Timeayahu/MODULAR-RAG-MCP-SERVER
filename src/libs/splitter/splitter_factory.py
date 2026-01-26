"""Splitter 工厂：根据配置创建对应的 Splitter 实例。"""

from typing import Dict, Type

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """Splitter 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseSplitter]] = {}

    @classmethod
    def register(cls, provider: str, splitter_cls: Type[BaseSplitter]) -> None:
        """注册 Splitter 实现。"""
        cls._registry[provider.lower()] = splitter_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """取消注册 Splitter 实现。"""
        cls._registry.pop(provider.lower(), None)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（测试辅助）。"""
        cls._registry.clear()

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """根据配置创建 Splitter 实例。

        Args:
            settings: 全局配置。

        Returns:
            BaseSplitter 实例。

        Raises:
            ValueError: provider 未注册。
        """
        provider = settings.splitter.provider.lower()
        if provider not in cls._registry:
            raise ValueError(f"未知的 Splitter provider: {provider}")
        return cls._registry[provider](settings.splitter)

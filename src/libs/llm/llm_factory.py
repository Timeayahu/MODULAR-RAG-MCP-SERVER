"""LLM 工厂：根据配置创建对应的 LLM 实例。"""

from typing import Dict, Type

from core.settings import Settings
from libs.llm.base_llm import BaseLLM


class LLMFactory:
    """LLM 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseLLM]] = {}

    @classmethod
    def register(cls, provider: str, llm_cls: Type[BaseLLM]) -> None:
        """注册 LLM 实现。

        Args:
            provider: provider 名称。
            llm_cls: LLM 实现类。
        """
        cls._registry[provider.lower()] = llm_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """取消注册 LLM 实现。"""
        cls._registry.pop(provider.lower(), None)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表（测试辅助）。"""
        cls._registry.clear()

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """根据配置创建 LLM 实例。

        Args:
            settings: 全局配置。

        Returns:
            BaseLLM 实例。

        Raises:
            ValueError: provider 未注册。
        """
        provider = settings.llm.provider.lower()
        if provider not in cls._registry:
            raise ValueError(f"未知的 LLM provider: {provider}")
        return cls._registry[provider](settings.llm)

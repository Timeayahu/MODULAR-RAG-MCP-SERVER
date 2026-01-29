"""Splitter 工厂：根据配置创建对应的 Splitter 实例。

根据 DEV_SPEC 3.3.3：
- 分块是 Ingestion Pipeline 的核心环节之一
- LlamaIndex Ingestion Pipeline 的 Splitter 环节支持可插拔设计
- 本项目当前采用 LangChain 的 RecursiveCharacterTextSplitter 进行切分

本工厂支持以下 Provider：
- recursive: LangChain RecursiveCharacterTextSplitter（默认）

架构设计上预留了切换能力，可扩展支持：
- semantic: LlamaIndex SemanticSplitter
- fixed: 固定长度切分器
"""

from typing import Dict, Type

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter


class SplitterFactory:
    """Splitter 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseSplitter]] = {"recursive": RecursiveSplitter}

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

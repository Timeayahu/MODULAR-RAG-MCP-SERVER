"""LLM 工厂：根据配置创建对应的 LLM 实例。

根据 DEV_SPEC 3.3.1 设计原则：
- 配置驱动 (Configuration-Driven)：通过统一配置文件指定各组件的具体后端
- 工厂模式 (Factory Pattern)：使用工厂函数根据配置动态实例化对应的实现类

本工厂基于 LlamaIndex LLM 抽象层，支持以下 Provider：
- openai: OpenAI 官方 API
- azure: Azure OpenAI
- ollama: Ollama 本地模型
- deepseek: DeepSeek API (OpenAI-Compatible)
"""

from typing import Dict, Type

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM


class LLMFactory:
    """LLM 工厂，按 provider 路由到具体实现。"""

    _registry: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "ollama": OllamaLLM,
    }

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

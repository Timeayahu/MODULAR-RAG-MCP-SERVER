"""OpenAI LLM 实现，基于 LlamaIndex OpenAI 适配器。

根据 DEV_SPEC 3.3.2：
- LlamaIndex 内置了对主流 LLM Provider 的适配（OpenAI、Azure、Ollama 等）
- 对于 LlamaIndex 未覆盖的 Provider，可通过其 OpenAI-Compatible 模式接入
"""

from typing import TYPE_CHECKING, Optional

from libs.llm.base_llm import BaseLLM, LLAMA_INDEX_AVAILABLE

if LLAMA_INDEX_AVAILABLE:
    from llama_index.core.llms import LLM as LlamaLLM

if TYPE_CHECKING:
    from llama_index.core.llms import LLM as LlamaLLM


class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现，基于 LlamaIndex OpenAI 适配器。"""

    provider_name: str = "openai"

    def get_llama_llm(self) -> "LlamaLLM":
        """获取 LlamaIndex OpenAI LLM 实例。"""
        if self._llama_llm is not None:
            return self._llama_llm

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.llms.openai import OpenAI
        except ImportError:
            raise ImportError(
                "llama-index-llms-openai 未安装，请运行: pip install llama-index-llms-openai"
            )

        # 构建参数
        kwargs = {
            "model": self.config.model,
            "api_key": self.config.api_key,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # 可选参数
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        self._llama_llm = OpenAI(**kwargs)
        return self._llama_llm


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-Compatible LLM 基类，用于支持 DeepSeek 等兼容接口。

    通过设置自定义 api_base 实现对 OpenAI-Compatible API 的支持。
    """

    provider_name: str = "openai_compatible"
    default_base_url: Optional[str] = None

    def get_llama_llm(self) -> "LlamaLLM":
        """获取 LlamaIndex OpenAI LLM 实例（使用自定义 base_url）。"""
        if self._llama_llm is not None:
            return self._llama_llm

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.llms.openai import OpenAI
        except ImportError:
            raise ImportError(
                "llama-index-llms-openai 未安装，请运行: pip install llama-index-llms-openai"
            )

        base_url = self.config.base_url or self.default_base_url
        if not base_url:
            raise ValueError(f"{self.provider_name} 缺少 base_url")

        kwargs = {
            "model": self.config.model,
            "api_key": self.config.api_key,
            "api_base": base_url,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        self._llama_llm = OpenAI(**kwargs)
        return self._llama_llm

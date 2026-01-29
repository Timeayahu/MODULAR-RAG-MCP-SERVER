"""Ollama LLM 实现，基于 LlamaIndex Ollama 适配器。

根据 DEV_SPEC 3.3.2：
- Ollama / vLLM (本地)：完全离线、隐私敏感、无 API 成本
- LlamaIndex 内置了对 Ollama 的适配
"""

from typing import TYPE_CHECKING

from libs.llm.base_llm import BaseLLM, LLAMA_INDEX_AVAILABLE

if TYPE_CHECKING:
    from llama_index.core.llms import LLM as LlamaLLM


class OllamaLLM(BaseLLM):
    """Ollama LLM 实现，基于 LlamaIndex Ollama 适配器。"""

    provider_name: str = "ollama"
    default_base_url: str = "http://localhost:11434"

    def get_llama_llm(self) -> "LlamaLLM":
        """获取 LlamaIndex Ollama LLM 实例。"""
        if self._llama_llm is not None:
            return self._llama_llm

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.llms.ollama import Ollama
        except ImportError:
            raise ImportError(
                "llama-index-llms-ollama 未安装，请运行: pip install llama-index-llms-ollama"
            )

        base_url = self.config.base_url or self.default_base_url

        kwargs = {
            "model": self.config.model,
            "base_url": base_url,
            "request_timeout": 120.0,  # Ollama 本地模型可能需要较长时间
        }

        # Ollama 不需要 api_key，但可以设置其他参数
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        self._llama_llm = Ollama(**kwargs)
        return self._llama_llm

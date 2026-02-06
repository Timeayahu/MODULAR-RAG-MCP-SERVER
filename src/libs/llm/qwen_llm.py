"""Qwen LLM 实现，基于 LlamaIndex DashScope LLM 适配器。

Qwen (DashScope) 提供原生 LLM API，LlamaIndex 有专门的适配器支持。
支持的模型：qwen-turbo, qwen-plus, qwen-max 等。
"""

from typing import TYPE_CHECKING

from libs.llm.base_llm import BaseLLM, LLAMA_INDEX_AVAILABLE

if TYPE_CHECKING:
    from llama_index.core.llms import LLM as LlamaLLM


class QwenLLM(BaseLLM):
    """Qwen LLM 实现，基于 LlamaIndex DashScope 适配器。"""

    provider_name: str = "qwen"

    def get_llama_llm(self) -> "LlamaLLM":
        """获取 LlamaIndex DashScope LLM 实例。"""
        if self._llama_llm is not None:
            return self._llama_llm

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
        except ImportError:
            raise ImportError(
                "llama-index-llms-dashscope 未安装，请运行: pip install llama-index-llms-dashscope"
            )

        # 构建参数
        kwargs = {
            "model_name": self.config.model,
            "api_key": self.config.api_key,
        }

        # 可选参数
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        self._llama_llm = DashScope(**kwargs)
        return self._llama_llm

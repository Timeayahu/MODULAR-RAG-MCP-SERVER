"""Azure OpenAI LLM 实现，基于 LlamaIndex Azure OpenAI 适配器。

根据 DEV_SPEC 3.3.2：
- Azure OpenAI：企业级 Azure 云端服务，符合合规与安全要求
"""

from typing import TYPE_CHECKING

from libs.llm.base_llm import BaseLLM, LLAMA_INDEX_AVAILABLE

if TYPE_CHECKING:
    from llama_index.core.llms import LLM as LlamaLLM


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM 实现，基于 LlamaIndex Azure OpenAI 适配器。"""

    provider_name: str = "azure"

    def get_llama_llm(self) -> "LlamaLLM":
        """获取 LlamaIndex Azure OpenAI LLM 实例。"""
        if self._llama_llm is not None:
            return self._llama_llm

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.llms.azure_openai import AzureOpenAI
        except ImportError:
            # 尝试从 openai 模块导入
            try:
                from llama_index.llms.openai import AzureOpenAI
            except ImportError:
                raise ImportError(
                    "llama-index-llms-openai 未安装，请运行: pip install llama-index-llms-openai"
                )

        # Azure 特有配置
        # base_url 格式: https://<resource>.openai.azure.com/
        azure_endpoint = self.config.base_url
        if not azure_endpoint:
            raise ValueError("Azure OpenAI 缺少 base_url (azure_endpoint)")

        kwargs = {
            "model": self.config.model,  # deployment_name
            "azure_endpoint": azure_endpoint,
            "api_key": self.config.api_key,
            "api_version": "2024-02-01",  # 可通过配置扩展
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        self._llama_llm = AzureOpenAI(**kwargs)
        return self._llama_llm

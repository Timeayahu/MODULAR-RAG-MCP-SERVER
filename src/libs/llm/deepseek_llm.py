"""DeepSeek LLM 实现，通过 OpenAI-Compatible 模式接入。

根据 DEV_SPEC 3.3.2：
- 对于 LlamaIndex 未覆盖的 Provider（如 DeepSeek），
  可通过其 OpenAI-Compatible 模式接入（设置自定义 api_base）
"""

from libs.llm.openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek LLM 实现，通过 OpenAI-Compatible 模式接入。"""

    provider_name: str = "deepseek"
    default_base_url: str = "https://api.deepseek.com/v1"

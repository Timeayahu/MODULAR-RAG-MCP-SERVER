"""Ollama LLM 实现。"""

from libs.llm.openai_llm import OpenAICompatibleLLM


class OllamaLLM(OpenAICompatibleLLM):
    """Ollama 本地 LLM 实现（OpenAI-Compatible）。"""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"
    auth_header_name = ""

"""
LLM 抽象

大语言模型抽象层，支持 OpenAI、Azure、Ollama、DeepSeek 等多种 LLM 提供商。

为什么需要__init__?
    init模块定义了可以被外部访问的模块，决定哪些可以被访问哪些不想被用户看到，并简化了导入路径
    init本身存在是为了告诉 Python，libs/llm 这个文件夹是一个可导入的包，而不是普通的文件夹
"""

from .base_llm import BaseLLM
from .llm_factory import LLMFactory

__all__ = ["BaseLLM", "LLMFactory"]
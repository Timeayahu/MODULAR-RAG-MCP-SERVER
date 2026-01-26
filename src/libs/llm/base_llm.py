"""LLM 抽象基类定义。"""

from abc import ABC, abstractmethod
from typing import Dict, List
from core.settings import LLMConfig


class BaseLLM(ABC):
    """LLM 抽象基类，定义统一的对话接口。"""

    def __init__(self, config: LLMConfig) -> None:
        """初始化 LLM。

        Args:
            config: LLM 配置。
        """
        self._config = config

    @property
    def config(self) -> LLMConfig:
        """返回 LLM 配置。"""
        return self._config

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """执行对话并返回模型输出。

        Args:
            messages: 对话消息列表，元素为 {"role": "...", "content": "..."}。

        Returns:
            模型输出文本。
        """
        raise NotImplementedError

"""LLM 抽象基类定义。

本模块基于 LlamaIndex 的 LLM 抽象进行封装，
提供与 LlamaIndex 生态无缝集成的 LLM 接口。

根据 DEV_SPEC 3.3.2：
- 本项目以 LlamaIndex 为主框架
- LlamaIndex 的 LLM 抽象类已封装了统一调用接口
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from core.settings import LLMConfig

# LlamaIndex LLM 类型导入
try:
    from llama_index.core.llms import LLM as LlamaLLM
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.core.llms import ChatResponse, CompletionResponse

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    LlamaLLM = None  # type: ignore
    ChatMessage = None  # type: ignore
    MessageRole = None  # type: ignore

if TYPE_CHECKING:
    from llama_index.core.llms import LLM as LlamaLLM


class BaseLLM(ABC):
    """LLM 抽象基类，基于 LlamaIndex LLM 接口设计。

    设计说明：
    - 提供与 LlamaIndex LLM 兼容的接口
    - 支持通过 get_llama_llm() 获取原生 LlamaIndex LLM 实例
    - 保持简洁的 chat() 方法用于快速调用
    """

    def __init__(self, config: LLMConfig) -> None:
        """初始化 LLM。

        Args:
            config: LLM 配置。
        """
        self._config = config
        self._llama_llm: Optional["LlamaLLM"] = None

    @property
    def config(self) -> LLMConfig:
        """返回 LLM 配置。"""
        return self._config

    @abstractmethod
    def get_llama_llm(self) -> "LlamaLLM":
        """获取 LlamaIndex 原生 LLM 实例。

        Returns:
            LlamaIndex LLM 实例。

        Raises:
            ImportError: 如果 llama-index 未安装。
        """
        raise NotImplementedError

    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """执行对话并返回模型输出。

        支持两种输入格式：
        1. 简单字符串：作为 user 消息发送
        2. 消息列表：[{"role": "user", "content": "..."}, ...]

        Args:
            messages: 对话消息，可以是字符串或消息列表。

        Returns:
            模型输出文本。
        """
        llm = self.get_llama_llm()

        # 转换为 LlamaIndex ChatMessage 格式
        chat_messages = self._convert_to_chat_messages(messages)

        # 调用 LlamaIndex LLM
        response = llm.chat(chat_messages)

        return response.message.content or ""

    def complete(self, prompt: str) -> str:
        """执行补全请求。

        Args:
            prompt: 补全提示词。

        Returns:
            模型输出文本。
        """
        llm = self.get_llama_llm()
        response = llm.complete(prompt)
        return response.text or ""

    def _convert_to_chat_messages(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> List["ChatMessage"]:
        """将输入转换为 LlamaIndex ChatMessage 列表。

        Args:
            messages: 输入消息。

        Returns:
            ChatMessage 列表。
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        if isinstance(messages, str):
            return [ChatMessage(role=MessageRole.USER, content=messages)]

        chat_messages = []
        for msg in messages:
            role_str = msg.get("role", "user").lower()
            content = msg.get("content", "")

            if role_str == "system":
                role = MessageRole.SYSTEM
            elif role_str == "assistant":
                role = MessageRole.ASSISTANT
            else:
                role = MessageRole.USER

            chat_messages.append(ChatMessage(role=role, content=content))

        return chat_messages

"""Splitter 抽象基类定义。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from core.settings import SplitterConfig

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """文本切分器抽象基类，定义统一的切分接口。"""

    def __init__(self, config: SplitterConfig) -> None:
        """初始化 Splitter。

        Args:
            config: Splitter 配置。
        """
        self._config = config

    @property
    def config(self) -> SplitterConfig:
        """返回 Splitter 配置。"""
        return self._config

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional["TraceContext"] = None,
    ) -> List[str]:
        """切分文本为多个片段。

        Args:
            text: 输入文本。
            trace: 可选追踪上下文。

        Returns:
            切分后的文本片段列表。
        """
        raise NotImplementedError

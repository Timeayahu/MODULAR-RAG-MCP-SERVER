"""Splitter 抽象基类定义。

根据 DEV_SPEC 3.1.1：
- Splitter 基于 Markdown 结构（标题/段落/代码块等）与参数配置把 Document 切为若干 Chunk
- 每个 chunk 必须携带稳定的定位信息与来源信息：source, chunk_index, start_offset/end_offset
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from core.settings import SplitterConfig

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass
class SplitResult:
    """切分结果，包含文本和定位信息。

    根据 DEV_SPEC 3.1.1 要求：
    - 每个 chunk 必须携带稳定的定位信息：chunk_index, start_offset, end_offset
    """

    text: str
    chunk_index: int
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSplitter(ABC):
    """文本切分器抽象基类，定义统一的切分接口。

    设计说明：
    - 提供两个切分方法：split_text() 返回字符串列表，split() 返回带定位信息的 SplitResult 列表
    - 子类必须实现 split() 方法，split_text() 有默认实现
    """

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
    def split(
        self,
        text: str,
        trace: Optional["TraceContext"] = None,
    ) -> List[SplitResult]:
        """切分文本为多个片段，返回带定位信息的结果。

        Args:
            text: 输入文本。
            trace: 可选追踪上下文。

        Returns:
            切分后的 SplitResult 列表，包含文本和定位信息。
        """
        raise NotImplementedError

    def split_text(
        self,
        text: str,
        trace: Optional["TraceContext"] = None,
    ) -> List[str]:
        """切分文本为多个片段（简化接口，仅返回文本列表）。

        Args:
            text: 输入文本。
            trace: 可选追踪上下文。

        Returns:
            切分后的文本片段列表。
        """
        results = self.split(text, trace)
        return [r.text for r in results]

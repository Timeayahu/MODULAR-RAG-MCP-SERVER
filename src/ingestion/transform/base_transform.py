"""Transform 抽象基类定义。

负责对切分后的 Chunk 进行语义增强、去噪或元数据注入。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from ingestion.models import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Chunk 转换器抽象基类。"""

    @abstractmethod
    def transform(
        self, chunks: List[Chunk], trace: Optional["TraceContext"] = None
    ) -> List[Chunk]:
        """对多个 Chunk 进行转换处理。

        Args:
            chunks: 输入的 Chunk 列表。
            trace: 可选的追踪上下文。

        Returns:
            转换后的 Chunk 列表。
        """
        raise NotImplementedError

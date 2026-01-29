"""Reranker 抽象基类定义。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.settings import RerankConfig

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseReranker(ABC):
    """重排序抽象基类，定义统一的重排接口。"""

    def __init__(self, config: RerankConfig) -> None:
        """初始化 Reranker。

        Args:
            config: 重排序配置。
        """
        self._config = config

    @property
    def config(self) -> RerankConfig:
        """返回重排序配置。"""
        return self._config

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, Any]]:
        """对候选结果进行重排序。

        Args:
            query: 用户查询。
            candidates: 候选列表。
            trace: 可选追踪上下文。

        Returns:
            重排后的候选列表。
        """
        raise NotImplementedError

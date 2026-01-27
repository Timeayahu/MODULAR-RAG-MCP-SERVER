"""Evaluator 抽象基类定义。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from core.settings import EvaluationConfig

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEvaluator(ABC):
    """评估器抽象基类，定义统一的评估接口。"""

    def __init__(self, config: EvaluationConfig) -> None:
        """初始化 Evaluator。

        Args:
            config: 评估配置。
        """
        self._config = config

    @property
    def config(self) -> EvaluationConfig:
        """返回评估配置。"""
        return self._config

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional["TraceContext"] = None,
    ) -> Dict[str, float]:
        """评估检索结果。

        Args:
            query: 用户查询。
            retrieved_ids: 检索命中的 id 列表（按排名顺序）。
            golden_ids: 标准答案 id 列表。
            trace: 可选追踪上下文。

        Returns:
            指标字典，如 hit_rate/mrr 等。
        """
        raise NotImplementedError

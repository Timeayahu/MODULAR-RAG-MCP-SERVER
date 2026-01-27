"""自定义轻量评估器实现。"""

from typing import TYPE_CHECKING, Dict, List, Optional

from libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class CustomEvaluator(BaseEvaluator):
    """最小自定义评估器，支持 hit_rate 与 mrr。"""

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional["TraceContext"] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        requested = [name.lower() for name in self.config.metrics]

        if "hit_rate" in requested:
            metrics["hit_rate"] = self._hit_rate(retrieved_ids, golden_ids)
        if "mrr" in requested:
            metrics["mrr"] = self._mrr(retrieved_ids, golden_ids)

        return metrics

    @staticmethod
    def _hit_rate(retrieved_ids: List[str], golden_ids: List[str]) -> float:
        if not golden_ids:
            return 0.0
        golden_set = set(golden_ids)
        return 1.0 if any(item in golden_set for item in retrieved_ids) else 0.0

    @staticmethod
    def _mrr(retrieved_ids: List[str], golden_ids: List[str]) -> float:
        if not golden_ids:
            return 0.0
        golden_set = set(golden_ids)
        for index, item in enumerate(retrieved_ids, start=1):
            if item in golden_set:
                return 1.0 / index
        return 0.0

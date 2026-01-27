"""Cross-Encoder Reranker 实现（占位）。"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from libs.reranker.base_reranker import BaseReranker

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class ScorerProtocol(Protocol):
    """评分器协议，用于注入 mock/scorer 实现。"""

    def score(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """返回与 candidates 等长的分数列表。"""


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排序实现（占位）。"""

    def __init__(
        self,
        config,
        scorer: Optional[ScorerProtocol] = None,
    ) -> None:
        super().__init__(config)
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query:
            raise ValueError("cross encoder query 不能为空")
        if not isinstance(candidates, list):
            raise ValueError("cross encoder candidates 格式错误")
        if not candidates:
            return []
        if self._scorer is None:
            raise ValueError("cross encoder scorer 未配置")

        scores = self._scorer.score(query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("cross encoder scores 长度不匹配")

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [item[0] for item in ranked[: self.config.top_k]]

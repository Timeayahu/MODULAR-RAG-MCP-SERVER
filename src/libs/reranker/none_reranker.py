"""None Reranker：不做重排序，直接返回原顺序。"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from libs.reranker.base_reranker import BaseReranker

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class NoneReranker(BaseReranker):
    """默认回退 Reranker，保持原顺序。"""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, Any]]:
        return candidates

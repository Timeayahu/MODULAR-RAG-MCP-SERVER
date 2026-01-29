"""LLM Reranker 实现。"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.settings import RerankConfig, Settings
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class LLMReranker(BaseReranker):
    """基于 LLM 的重排序实现。"""

    default_prompt_path = Path("config/prompts/rerank.txt")

    def __init__(
        self,
        config: RerankConfig,
        llm,
        prompt_text: Optional[str] = None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        super().__init__(config)
        self._llm = llm
        self._prompt_text = prompt_text
        self._prompt_path = prompt_path or self.default_prompt_path

    @classmethod
    def from_settings(
        cls, settings: Settings, prompt_text: Optional[str] = None
    ) -> "LLMReranker":
        """从全局配置创建 LLMReranker。"""
        llm = LLMFactory.create(settings)
        return cls(settings.rerank, llm, prompt_text=prompt_text)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query:
            raise ValueError("llm reranker query 不能为空")
        if not isinstance(candidates, list):
            raise ValueError("llm reranker candidates 格式错误")
        if not candidates:
            return []

        prompt = self._build_prompt(query, candidates)
        response = self._llm.chat([{"role": "user", "content": prompt}])
        ranked_ids = self._parse_ranked_ids(response)
        return self._reorder_candidates(candidates, ranked_ids)

    def _build_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        prompt_template = self._prompt_text or self._prompt_path.read_text(
            encoding="utf-8"
        )
        chunks_text = []
        for item in candidates:
            item_id = item.get("id")
            text = item.get("text", "")
            if not isinstance(item_id, str):
                raise ValueError("llm reranker candidates 缺少 id")
            chunks_text.append(f"- id: {item_id}\n  text: {text}")
        return prompt_template.format(query=query, chunks="\n".join(chunks_text))

    def _parse_ranked_ids(self, response: str) -> List[str]:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("llm reranker 输出格式错误") from exc
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("llm reranker 输出为空")
        if not all(isinstance(item, str) for item in parsed):
            raise ValueError("llm reranker 输出格式错误")
        return parsed

    def _reorder_candidates(
        self, candidates: List[Dict[str, Any]], ranked_ids: List[str]
    ) -> List[Dict[str, Any]]:
        candidate_map = {item.get("id"): item for item in candidates}
        ordered: List[Dict[str, Any]] = []
        for item_id in ranked_ids:
            if item_id not in candidate_map:
                raise ValueError("llm reranker 输出包含未知 id")
            ordered.append(candidate_map[item_id])
            if len(ordered) >= self.config.top_k:
                break
        return ordered

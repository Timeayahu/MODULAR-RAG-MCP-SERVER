"""Chunk 智能重组与去噪模块。

负责对原始分块进行二次加工：
1. 规则去噪：去除多余空白、页眉页脚占位符等。
2. (可选) LLM 重写：利用 LLM 提高文本连贯性与语义完整性。
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ingestion.models import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.llm_factory import LLMFactory
from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class ChunkRefiner(BaseTransform):
    """Chunk 精炼器实现。"""

    def __init__(self, settings: Settings) -> None:
        """初始化 ChunkRefiner。

        Args:
            settings: 全局配置。
        """
        self._settings = settings
        self._config = settings.transform
        self._llm = None
        if self._config.refine_use_llm:
            try:
                self._llm = LLMFactory.create(settings)
            except Exception as e:
                logger.warning(f"无法初始化 LLM 用于 ChunkRefiner: {e}. 将回退到规则模式。")

        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """从配置文件加载 Prompt 模板。"""
        prompt_path = Path("config/prompts/chunk_refinement.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "请精炼以下文本块：\n\n{text}"

    def transform(
        self, chunks: List[Chunk], trace: Optional["TraceContext"] = None
    ) -> List[Chunk]:
        """对 Chunk 列表执行精炼处理。

        Args:
            chunks: 输入的 Chunk 列表。
            trace: 可选的追踪上下文。

        Returns:
            处理后的 Chunk 列表。
        """
        if not self._config.refine_enabled:
            return chunks

        refined_chunks = []
        for chunk in chunks:
            # 1. 规则去噪 (总是执行)
            cleaned_text = self._rule_based_cleanup(chunk.text)
            
            # 2. (可选) LLM 重写
            final_text = cleaned_text
            if self._config.refine_use_llm and self._llm:
                try:
                    final_text = self._llm_refinement(cleaned_text, trace)
                except Exception as e:
                    logger.error(f"LLM 精炼失败 (chunk_id={chunk.id}): {e}. 降级使用规则去噪结果。")
                    chunk.metadata["refine_fallback"] = True
            
            chunk.text = final_text
            refined_chunks.append(chunk)
            
        return refined_chunks

    def _rule_based_cleanup(self, text: str) -> str:
        """基于规则的基础清洗。"""
        if not text:
            return ""
        
        # 1. 去除每一行行首行尾的空白
        lines = [line.strip() for line in text.splitlines()]
        
        # 2. 去除多余空行，保持最多一个空行间隔
        cleaned_lines = []
        last_was_empty = False
        for line in lines:
            if not line:
                if not last_was_empty:
                    cleaned_lines.append("")
                    last_was_empty = True
            else:
                cleaned_lines.append(line)
                last_was_empty = False
        
        text = "\n".join(cleaned_lines).strip()
        
        # 3. 示例规则：去除常见的页码模式 (例如 "Page 1 of 10")
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE).strip()
        
        return text

    def _llm_refinement(self, text: str, trace: Optional["TraceContext"] = None) -> str:
        """使用 LLM 执行深度精炼。"""
        if not self._llm:
            return text
            
        prompt = f"{self._prompt_template}\n\n{text}"
        # 这里为了简化，直接调用 chat。实际可能需要更复杂的消息构造。
        response = self._llm.chat(prompt)
        return response.strip()

"""元数据增强模块。

负责为 Chunk 提取并注入更丰富的语义元数据：
1. 规则增强：从文本特征推断标题、生成简单摘要。
2. (可选) LLM 增强：利用 LLM 提取高维语义特征 (Title, Summary, Tags)。
"""

import logging
import json
from typing import List, Optional, TYPE_CHECKING

from ingestion.models import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.llm_factory import LLMFactory
from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransform):
    """元数据增强器实现。"""

    def __init__(self, settings: Settings) -> None:
        """初始化 MetadataEnricher。"""
        self._settings = settings
        self._config = settings.transform
        self._llm = None
        if self._config.enrich_metadata and getattr(self._config, "enrich_use_llm", False):
            try:
                self._llm = LLMFactory.create(settings)
            except Exception as e:
                logger.warning(f"无法初始化 LLM 用于 MetadataEnricher: {e}. 将回退到规则模式。")

    def transform(
        self, chunks: List[Chunk], trace: Optional["TraceContext"] = None
    ) -> List[Chunk]:
        """对 Chunk 列表执行元数据增强。"""
        if not self._config.enrich_metadata:
            return chunks

        enriched_chunks = []
        for chunk in chunks:
            # 1. 规则模式 (基础)
            self._rule_based_enrichment(chunk)
            
            # 2. (可选) LLM 模式
            if getattr(self._config, "enrich_use_llm", False) and self._llm:
                try:
                    self._llm_based_enrichment(chunk, trace)
                except Exception as e:
                    logger.error(f"LLM 元数据增强失败 (chunk_id={chunk.id}): {e}. 保留规则模式结果。")
                    chunk.metadata["enrich_fallback"] = True
            
            enriched_chunks.append(chunk)
            
        return enriched_chunks

    def _rule_based_enrichment(self, chunk: Chunk) -> None:
        """基于规则的提取：标题、初步摘要。"""
        text = chunk.text.strip()
        if not text:
            return

        # 提取第一行作为默认标题（若元数据中还没有）
        if "title" not in chunk.metadata:
            first_line = text.split('\n')[0][:100]  # 限制长度
            chunk.metadata["title"] = first_line.strip("# ").strip()

        # 生成简单摘要（前 150 字符）
        if "summary" not in chunk.metadata:
            chunk.metadata["summary"] = text[:150].replace('\n', ' ') + "..."

        # 默认标签
        if "tags" not in chunk.metadata:
            chunk.metadata["tags"] = []

    def _llm_based_enrichment(self, chunk: Chunk, trace: Optional["TraceContext"] = None) -> None:
        """使用 LLM 提取结构化元数据。"""
        prompt = (
            "分析以下文本块，并以 JSON 格式输出其元数据：\n"
            "{\n"
            "  \"title\": \"精准的短标题\",\n"
            "  \"summary\": \"一句话内容摘要\",\n"
            "  \"tags\": [\"标签1\", \"标签2\"]\n"
            "}\n\n"
            f"文本内容：\n{chunk.text}"
        )
        
        response = self._llm.chat(prompt)
        try:
            # 尝试解析 JSON（处理 LLM 可能返回的 Markdown 代码块包装）
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif json_str.startswith("```"):
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            chunk.metadata["title"] = data.get("title", chunk.metadata.get("title"))
            chunk.metadata["summary"] = data.get("summary", chunk.metadata.get("summary"))
            chunk.metadata["tags"] = list(set(chunk.metadata.get("tags", []) + data.get("tags", [])))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"解析 LLM 元数据 JSON 失败: {e}. 响应内容: {response}")
            raise e

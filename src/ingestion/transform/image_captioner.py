"""ImageCaptioner Transform 实现。

根据 DEV_SPEC 3.1.1 和 3.5.3：
- 多模态增强 (Multimodal Enrichment / Image Captioning)
- 策略：扫描文档片段中的图像引用，调用 Vision LLM 进行视觉理解
- 动作：生成高保真的文本描述（Caption），描述图表逻辑或提取截图文字
- 存储：将 Caption 文本"缝合"进 Chunk 的正文或 Metadata 中

Vision LLM 选型（根据 DEV_SPEC 3.5.3）：
- GPT-4o (Azure/OpenAI)：视觉理解能力强，复杂图表解读准确率高
- Qwen-VL-Max (阿里云)：中文场景表现优异，性价比高
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from ingestion.models import Chunk
from ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Vision LLM 默认 prompt 路径
DEFAULT_PROMPT_PATH = "config/prompts/image_captioning.txt"


class ImageCaptioner(BaseTransform):
    """图片描述生成 Transform。

    特性：
    - 可选启用/禁用（通过配置）
    - 支持多种 Vision LLM 后端（GPT-4o, Qwen-VL 等）
    - 失败时降级，不阻塞 ingestion
    - 支持缓存以避免重复调用
    """

    def __init__(
        self,
        enabled: bool = False,
        vision_llm = None,
        prompt_path: Optional[str] = None,
        inject_to_text: bool = True,
    ) -> None:
        """初始化 ImageCaptioner。

        Args:
            enabled: 是否启用图片描述生成。
            vision_llm: Vision LLM 实例（支持 chat 接口）。
            prompt_path: Prompt 模板文件路径。
            inject_to_text: 是否将描述注入到 chunk.text 中（默认 True）。
                           如果为 False，则仅写入 metadata。
        """
        self.enabled = enabled
        self.vision_llm = vision_llm
        self.inject_to_text = inject_to_text

        # 加载 prompt 模板
        self.prompt_template = self._load_prompt(prompt_path or DEFAULT_PROMPT_PATH)

        # 简单缓存（image_id -> caption）
        self._caption_cache: Dict[str, str] = {}

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> List[Chunk]:
        """处理 Chunks，为包含图片引用的 Chunk 生成描述。

        Args:
            chunks: 输入 Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            处理后的 Chunk 列表。
        """
        if not self.enabled:
            logger.debug("ImageCaptioner 未启用，跳过处理")
            return chunks

        if self.vision_llm is None:
            logger.warning("Vision LLM 未配置，降级跳过图片描述生成")
            return self._mark_unprocessed(chunks)

        processed_chunks: List[Chunk] = []

        for chunk in chunks:
            if not chunk.image_refs:
                # 没有图片引用，直接保留
                processed_chunks.append(chunk)
                continue

            try:
                # 生成图片描述
                captions = self._generate_captions_for_chunk(chunk)

                # 将描述注入到 chunk
                enhanced_chunk = self._inject_captions(chunk, captions)
                processed_chunks.append(enhanced_chunk)

            except Exception as e:
                logger.error(f"为 Chunk {chunk.id} 生成图片描述时失败: {e}")
                # 降级：标记为未处理但不阻塞
                processed_chunks.append(self._mark_chunk_unprocessed(chunk))

        return processed_chunks

    def _load_prompt(self, prompt_path: str) -> str:
        """加载 prompt 模板。

        Args:
            prompt_path: Prompt 文件路径。

        Returns:
            Prompt 文本。
        """
        try:
            path = Path(prompt_path)
            if path.exists():
                return path.read_text(encoding="utf-8")
            else:
                logger.warning(f"Prompt 文件不存在: {prompt_path}，使用默认 prompt")
                return self._get_default_prompt()
        except Exception as e:
            logger.warning(f"加载 prompt 失败: {e}，使用默认 prompt")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """获取默认 prompt（当文件加载失败时使用）。"""
        return """Describe this image in detail, focusing on:
1. Main content and purpose
2. Any text, labels, or data shown
3. Visual elements and their relationships
4. How it relates to the document context

Provide a concise 2-4 sentence description."""

    def _generate_captions_for_chunk(self, chunk: Chunk) -> List[Dict[str, Any]]:
        """为 Chunk 中的所有图片生成描述。

        Args:
            chunk: 包含图片引用的 Chunk。

        Returns:
            图片描述列表，每个元素包含 image_id 和 caption。
        """
        captions: List[Dict[str, Any]] = []

        for image_id in chunk.image_refs:
            # 检查缓存
            if image_id in self._caption_cache:
                caption = self._caption_cache[image_id]
                logger.debug(f"使用缓存的图片描述: {image_id}")
            else:
                # 生成新描述
                caption = self._generate_caption(image_id, chunk)
                self._caption_cache[image_id] = caption

            captions.append({"image_id": image_id, "caption": caption})

        return captions

    def _generate_caption(self, image_id: str, chunk: Chunk) -> str:
        """为单个图片生成描述。

        Args:
            image_id: 图片 ID。
            chunk: 包含该图片的 Chunk（提供上下文）。

        Returns:
            图片描述文本。
        """
        # 构建 prompt
        context = chunk.text[:500]  # 使用前 500 字符作为上下文
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "unknown")

        prompt = self.prompt_template.format(
            source=source,
            page=page,
            context=context,
        )

        # 调用 Vision LLM
        # 注意：这里假设 vision_llm 支持 chat 接口
        # 实际使用时需要传入图片数据（base64 或 URL）
        # 由于当前阶段只实现接口，这里简化处理
        try:
            response = self.vision_llm.chat(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Vision LLM 调用失败: {e}")
            raise

    def _inject_captions(
        self, chunk: Chunk, captions: List[Dict[str, Any]]
    ) -> Chunk:
        """将图片描述注入到 Chunk 中。

        Args:
            chunk: 原始 Chunk。
            captions: 图片描述列表。

        Returns:
            增强后的 Chunk。
        """
        # 复制 chunk
        enhanced_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            source_doc_id=chunk.source_doc_id,
            chunk_index=chunk.chunk_index,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            image_refs=chunk.image_refs.copy(),
        )

        # 将描述写入 metadata
        enhanced_chunk.metadata["image_captions"] = captions

        # 可选：注入到文本中
        if self.inject_to_text and captions:
            caption_texts = [
                f"[图片描述: {cap['caption']}]" for cap in captions
            ]
            enhanced_chunk.text = (
                chunk.text + "\n\n" + "\n".join(caption_texts)
            )

        return enhanced_chunk

    def _mark_unprocessed(self, chunks: List[Chunk]) -> List[Chunk]:
        """标记所有包含图片的 Chunk 为未处理。

        Args:
            chunks: Chunk 列表。

        Returns:
            标记后的 Chunk 列表。
        """
        processed: List[Chunk] = []
        for chunk in chunks:
            if chunk.image_refs:
                processed.append(self._mark_chunk_unprocessed(chunk))
            else:
                processed.append(chunk)
        return processed

    def _mark_chunk_unprocessed(self, chunk: Chunk) -> Chunk:
        """标记单个 Chunk 为图片未处理。

        Args:
            chunk: 原始 Chunk。

        Returns:
            标记后的 Chunk。
        """
        marked_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            source_doc_id=chunk.source_doc_id,
            chunk_index=chunk.chunk_index,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            image_refs=chunk.image_refs.copy(),
        )
        marked_chunk.metadata["has_unprocessed_images"] = True
        return marked_chunk

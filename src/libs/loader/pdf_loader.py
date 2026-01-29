"""PDF Loader 实现，基于 MarkItDown 进行 PDF→Markdown 转换。

根据 DEV_SPEC 3.1.1：
- 首选：MarkItDown（作为默认 PDF 解析/转换引擎）
- 优点是直接产出 Markdown 形态文本，便于与后续 RecursiveCharacterTextSplitter 的 separators 配合
- 输出标准 Document：id|source|text(markdown)|metadata

Loader 职责：
- 把原始文件解析为统一的 Document 对象（text + metadata）
- 统一输出格式采用规范化 Markdown 作为 Document.text
- 抽取/补齐基础 metadata（source_path, doc_type, title, images 引用列表等）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from ingestion.models import Document

from .base_loader import BaseLoader, _build_document_id

logger = logging.getLogger(__name__)

# MarkItDown 依赖
try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    MarkItDown = None  # type: ignore


class PdfLoader(BaseLoader):
    """PDF Loader 实现，基于 MarkItDown 进行 PDF→Markdown 转换。

    特性：
    - 使用 MarkItDown 将 PDF 转换为 Markdown 格式
    - 支持提取基础元数据（标题、页数等）
    - 提供降级方案（当 MarkItDown 不可用时）
    """

    DOC_TYPE = "pdf"

    def __init__(self) -> None:
        """初始化 PdfLoader。"""
        self._markitdown = None
        if MARKITDOWN_AVAILABLE:
            try:
                self._markitdown = MarkItDown()
            except Exception as e:
                logger.warning(f"MarkItDown 初始化失败: {e}. 将使用降级方案。")

    def load(self, path: str) -> Document:
        """加载 PDF 文件并转换为 Document。

        Args:
            path: PDF 文件路径。

        Returns:
            Document 对象，text 为 Markdown 格式。

        Raises:
            FileNotFoundError: 文件不存在。
            RuntimeError: 解析失败。
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # 使用 MarkItDown 转换
        if self._markitdown is not None:
            text, extra_metadata = self._convert_with_markitdown(file_path)
        else:
            # 降级方案：尝试直接读取（适用于文本 PDF 或测试）
            text, extra_metadata = self._fallback_read(file_path)

        doc_id = _build_document_id(file_path)

        # 构建元数据
        metadata: Dict[str, Any] = {
            BaseLoader.META_SOURCE_PATH: str(file_path.resolve()),
            BaseLoader.META_DOC_TYPE: self.DOC_TYPE,
        }
        metadata.update(extra_metadata)

        # 提取图片引用（如果有）
        images = self._extract_image_references(text)

        return Document(
            id=doc_id,
            text=text,
            metadata=metadata,
            images=images,
        )

    def _convert_with_markitdown(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """使用 MarkItDown 转换 PDF 为 Markdown。

        Args:
            file_path: PDF 文件路径。

        Returns:
            (markdown_text, extra_metadata) 元组。
        """
        try:
            result = self._markitdown.convert(str(file_path))
            text = result.text_content or ""
            
            # 提取可能的元数据
            extra_metadata: Dict[str, Any] = {}
            
            # 尝试从文本提取标题（第一个 # 开头的行）
            title = self._extract_title_from_markdown(text)
            if title:
                extra_metadata["title"] = title

            return text, extra_metadata

        except Exception as e:
            logger.error(f"MarkItDown 转换失败: {e}")
            # 尝试降级方案
            return self._fallback_read(file_path)

    def _fallback_read(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """降级方案：直接读取文件内容。

        适用于：
        - 测试环境（伪 PDF 文件）
        - 纯文本 PDF
        - MarkItDown 不可用时

        Args:
            file_path: 文件路径。

        Returns:
            (text, extra_metadata) 元组。
        """
        try:
            text = file_path.read_text(encoding="utf-8")
            logger.info(f"使用降级方案读取文件: {file_path}")
            return text, {"_loader_fallback": True}
        except UnicodeDecodeError:
            # 可能是真正的二进制 PDF
            raise RuntimeError(
                f"无法读取文件内容。请确保 MarkItDown 已安装: pip install markitdown\n"
                f"文件路径: {file_path}"
            )

    def _extract_title_from_markdown(self, text: str) -> str:
        """从 Markdown 文本提取标题。

        Args:
            text: Markdown 文本。

        Returns:
            标题字符串，如果未找到则返回空字符串。
        """
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            elif line.startswith("## "):
                return line[3:].strip()
        return ""

    def _extract_image_references(self, text: str) -> List[Dict[str, Any]]:
        """从 Markdown 文本提取图片引用。

        Markdown 图片语法: ![alt](url)

        Args:
            text: Markdown 文本。

        Returns:
            图片引用列表，每个元素包含 image_id, alt, url 等。
        """
        import re

        images: List[Dict[str, Any]] = []
        # Markdown 图片正则
        pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

        for idx, match in enumerate(re.finditer(pattern, text)):
            alt_text = match.group(1)
            url = match.group(2)
            images.append(
                {
                    "image_id": f"img_{idx}",
                    "alt": alt_text,
                    "url": url,
                    "position": match.start(),
                }
            )

        return images


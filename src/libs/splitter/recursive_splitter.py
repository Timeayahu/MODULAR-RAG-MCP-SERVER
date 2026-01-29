"""Recursive Splitter 实现，基于 LangChain RecursiveCharacterTextSplitter。

根据 DEV_SPEC 3.1.1 和 3.3.3：
- 使用 LangChain 的 RecursiveCharacterTextSplitter 进行切分
- 该方法对 Markdown 文档的结构（标题、段落、列表、代码块）有天然的适配性
- 能够通过配置语义断点（Separators）实现高质量、语义完整的切块
"""

from typing import TYPE_CHECKING, List, Optional

from libs.splitter.base_splitter import BaseSplitter, SplitResult

# LangChain 依赖
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

# Markdown 文档的默认分隔符，按优先级从高到低
# 参考 LangChain 的 MarkdownTextSplitter
MARKDOWN_SEPARATORS = [
    "\n## ",  # 二级标题
    "\n### ",  # 三级标题
    "\n#### ",  # 四级标题
    "\n##### ",  # 五级标题
    "\n###### ",  # 六级标题
    "\n\n",  # 段落分隔
    "\n",  # 换行
    " ",  # 空格
    "",  # 字符
]


class RecursiveSplitter(BaseSplitter):
    """递归字符切分器，基于 LangChain RecursiveCharacterTextSplitter。

    特性：
    - 按层级分隔符（段落→句子→字符）递归切分
    - 在长度限制内尽量保持语义边界
    - 对 Markdown 结构有天然适配性
    """

    def split(
        self,
        text: str,
        trace: Optional["TraceContext"] = None,
    ) -> List[SplitResult]:
        """切分文本为多个片段，返回带定位信息的结果。

        Args:
            text: 输入文本（推荐 Markdown 格式）。
            trace: 可选追踪上下文。

        Returns:
            切分后的 SplitResult 列表。
        """
        if text is None:
            raise ValueError("recursive splitter text 不能为空")
        if not isinstance(text, str):
            raise ValueError("recursive splitter text 格式错误")
        if text == "":
            return []

        chunk_size = max(1, self.config.chunk_size)
        overlap = max(0, self.config.chunk_overlap)
        if overlap >= chunk_size:
            raise ValueError("recursive splitter chunk_overlap 必须小于 chunk_size")

        if not LANGCHAIN_AVAILABLE:
            # 降级到简单实现
            return self._fallback_split(text, chunk_size, overlap)

        # 使用配置的分隔符或默认 Markdown 分隔符
        separators = self.config.separators or MARKDOWN_SEPARATORS

        # 创建 LangChain RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        # 使用 create_documents 以获取更多元数据
        # 或直接使用 split_text
        chunks = splitter.split_text(text)

        # 计算每个 chunk 的位置信息
        results: List[SplitResult] = []
        current_pos = 0
        for idx, chunk_text in enumerate(chunks):
            # 在原文中查找 chunk 的位置
            # 注意：由于 overlap，同一段文本可能出现多次
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                # 如果找不到（可能是因为 trim），从当前位置开始
                start_pos = current_pos
            end_pos = start_pos + len(chunk_text)

            results.append(
                SplitResult(
                    text=chunk_text,
                    chunk_index=idx,
                    start_offset=start_pos,
                    end_offset=end_pos,
                )
            )

            # 更新搜索起点（考虑 overlap）
            current_pos = max(current_pos, start_pos + 1)

        return results

    def _fallback_split(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[SplitResult]:
        """降级实现：当 LangChain 不可用时的简单切分。

        警告：此方法不保证语义完整性，仅作为备用方案。
        """
        step = chunk_size - overlap
        results: List[SplitResult] = []
        chunk_index = 0

        for start in range(0, len(text), step):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text:
                results.append(
                    SplitResult(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_offset=start,
                        end_offset=end,
                    )
                )
                chunk_index += 1

        return results

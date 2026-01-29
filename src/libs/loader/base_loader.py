"""Loader 抽象基类定义。

当前阶段只定义最小契约：
- `load(path) -> Document`：把单个文件解析为标准 `Document`。

具体格式解析逻辑由子类实现（例如 `PdfLoader`）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from ingestion.models import Document


class BaseLoader(ABC):
    """文档 Loader 抽象基类。

    约定：
    - 子类负责把给定路径的文件解析为 `Document`；
    - `Document.id` 默认使用文件的绝对路径字符串，保证在单机场景下全局唯一且可追溯；
    - `metadata` 至少包含：
        - `source_path`: 文件的绝对路径字符串
        - `doc_type`: 文档类型（例如 "pdf"、"txt" 等），由子类确定
    """

    #: metadata 字段名常量，便于调用方复用，避免硬编码字符串。
    META_SOURCE_PATH: Final[str] = "source_path"
    META_DOC_TYPE: Final[str] = "doc_type"

    @abstractmethod
    def load(self, path: str) -> Document:
        """加载并解析单个文档。

        :param path: 文件路径（相对或绝对路径）
        :return: 标准化后的 `Document` 对象
        """


def _build_document_id(path: Path) -> str:
    """根据文件路径生成稳定的 Document ID。

    目前策略：使用绝对路径字符串。
    """

    return str(path.resolve())


"""Ingestion 核心数据模型。

本模块基于 LlamaIndex 的 Document 和 TextNode 进行封装，
提供与 LlamaIndex 生态无缝集成的数据结构。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# LlamaIndex 核心类型导入
try:
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.schema import TextNode as LlamaTextNode
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    LlamaDocument = None  # type: ignore
    LlamaTextNode = None  # type: ignore


@dataclass
class Document:
    """原始文档模型，与 LlamaIndex Document 兼容。

    Attributes:
        id: 文档唯一标识。
        text: 文档文本内容（Markdown 格式）。
        metadata: 元数据字典，至少包含 source_path, doc_type。
        images: 图片引用列表（用于多模态处理）。
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict。"""
        return asdict(self)

    def to_llama_document(self) -> "LlamaDocument":
        """转换为 LlamaIndex Document 对象。

        Returns:
            LlamaIndex Document 实例。

        Raises:
            ImportError: 如果 llama-index 未安装。
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装，无法转换为 LlamaIndex Document")

        return LlamaDocument(
            doc_id=self.id,
            text=self.text,
            metadata={
                **self.metadata,
                "images": self.images,
            },
        )

    @classmethod
    def from_llama_document(cls, llama_doc: "LlamaDocument") -> "Document":
        """从 LlamaIndex Document 创建实例。

        Args:
            llama_doc: LlamaIndex Document 对象。

        Returns:
            Document 实例。
        """
        metadata = dict(llama_doc.metadata) if llama_doc.metadata else {}
        images = metadata.pop("images", [])

        return cls(
            id=llama_doc.doc_id or llama_doc.id_,
            text=llama_doc.text or "",
            metadata=metadata,
            images=images if isinstance(images, list) else [],
        )


@dataclass
class Chunk:
    """切分后的文本块模型，与 LlamaIndex TextNode 兼容。

    Attributes:
        id: Chunk 唯一标识。
        text: Chunk 文本内容。
        metadata: 元数据字典。
        source_doc_id: 来源文档 ID。
        chunk_index: 在文档中的切分索引。
        start_offset: 在原文档中的起始字符位置。
        end_offset: 在原文档中的结束字符位置。
        image_refs: 关联的图片 ID 列表。
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: Optional[str] = None
    chunk_index: int = 0
    start_offset: int = 0
    end_offset: int = 0
    image_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict。"""
        return asdict(self)

    def to_llama_node(self) -> "LlamaTextNode":
        """转换为 LlamaIndex TextNode 对象。

        Returns:
            LlamaIndex TextNode 实例。

        Raises:
            ImportError: 如果 llama-index 未安装。
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装，无法转换为 LlamaIndex TextNode")

        node = LlamaTextNode(
            id_=self.id,
            text=self.text,
            metadata={
                **self.metadata,
                "chunk_index": self.chunk_index,
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
                "image_refs": self.image_refs,
            },
            start_char_idx=self.start_offset,
            end_char_idx=self.end_offset,
        )

        # 设置来源文档关系
        if self.source_doc_id:
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=self.source_doc_id
            )

        return node

    @classmethod
    def from_llama_node(cls, node: "LlamaTextNode") -> "Chunk":
        """从 LlamaIndex TextNode 创建实例。

        Args:
            node: LlamaIndex TextNode 对象。

        Returns:
            Chunk 实例。
        """
        metadata = dict(node.metadata) if node.metadata else {}

        # 提取特殊字段
        chunk_index = metadata.pop("chunk_index", 0)
        start_offset = metadata.pop("start_offset", node.start_char_idx or 0)
        end_offset = metadata.pop("end_offset", node.end_char_idx or 0)
        image_refs = metadata.pop("image_refs", [])

        # 获取来源文档 ID
        source_doc_id = None
        if NodeRelationship.SOURCE in node.relationships:
            source_doc_id = node.relationships[NodeRelationship.SOURCE].node_id

        return cls(
            id=node.id_,
            text=node.text or "",
            metadata=metadata,
            source_doc_id=source_doc_id,
            chunk_index=chunk_index,
            start_offset=start_offset,
            end_offset=end_offset,
            image_refs=image_refs if isinstance(image_refs, list) else [],
        )

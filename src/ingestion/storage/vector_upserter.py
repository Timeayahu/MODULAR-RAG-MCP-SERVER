"""VectorUpserter 实现 - 向量存储器。

根据 DEV_SPEC 3.1.1 Upsert & Storage 阶段：
- 幂等性设计 (Idempotency)：为每个 Chunk 生成全局唯一的 chunk_id，
  生成算法采用确定的哈希组合：hash(source_path + section_path + content_hash)。
- 写入时采用 "Upsert"（更新或插入）语义，确保同一文档即使被多次处理，
  数据库中也永远只有一份最新副本，彻底避免重复索引问题。
- All-in-One 存储策略：每条记录同时包含 Index Data (Dense Vector, Sparse Vector)
  和 Payload Data (Content + Metadata)。
"""

import hashlib
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from ingestion.models import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext
    from libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorUpserter:
    """向量存储器。

    特性：
    - 生成稳定的 chunk_id（基于内容哈希）
    - 幂等 upsert（相同内容产生相同 id）
    - 同时存储 dense 和 sparse 向量
    - 存储完整的 metadata 和 content
    """

    def __init__(self, vector_store: "BaseVectorStore") -> None:
        """初始化 VectorUpserter。

        Args:
            vector_store: BaseVectorStore 实例。
        """
        self.vector_store = vector_store

    def upsert(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        trace: Optional["TraceContext"] = None,
    ) -> List[str]:
        """将 chunks 和对应的向量存储到向量数据库。

        Args:
            chunks: Chunk 列表。
            dense_vectors: 稠密向量列表。
            sparse_vectors: 稀疏向量列表。
            trace: 可选追踪上下文。

        Returns:
            生成的 chunk_id 列表。

        Raises:
            ValueError: 如果输入列表长度不一致。
        """
        if not chunks:
            raise ValueError("chunks 不能为空")

        # 验证输入长度一致
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 dense_vectors 数量 "
                f"({len(dense_vectors)}) 不一致"
            )

        if len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 sparse_vectors 数量 "
                f"({len(sparse_vectors)}) 不一致"
            )

        # 为每个 chunk 生成 chunk_id
        chunk_ids = []
        records = []

        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            # 生成稳定的 chunk_id
            chunk_id = self._generate_chunk_id(chunk)
            chunk_ids.append(chunk_id)

            # 基础 metadata（复制一份，避免修改原对象）
            metadata = chunk.metadata.copy()

            # 将稀疏向量和结构化字段放入 metadata，满足 All-in-One 存储策略
            if sparse_vec:
                metadata["sparse_vector"] = sparse_vec

            metadata["chunk_index"] = chunk.chunk_index
            metadata["source_doc_id"] = chunk.source_doc_id
            if chunk.start_offset is not None:
                metadata["start_offset"] = chunk.start_offset
            if chunk.end_offset is not None:
                metadata["end_offset"] = chunk.end_offset
            if chunk.image_refs:
                metadata["image_refs"] = chunk.image_refs

            # 构建存储记录，遵循 BaseVectorStore 的契约：
            # - id: 记录唯一标识
            # - vector: 稠密向量
            # - text: 原始文本
            # - metadata: 元数据
            record = {
                "id": chunk_id,
                "vector": dense_vec,
                "text": chunk.text,
                "metadata": metadata,
            }

            records.append(record)

        # 批量 upsert 到向量数据库
        try:
            self.vector_store.upsert(records, trace=trace)
            logger.info(f"成功 upsert {len(records)} 条记录")
        except Exception as e:
            logger.error(f"Upsert 失败: {e}")
            raise

        return chunk_ids

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """生成稳定的 chunk_id。

        算法：hash(source_path + section_path + content_hash)

        Args:
            chunk: Chunk 对象。

        Returns:
            稳定的 chunk_id（SHA256 哈希）。
        """
        # 获取 source_path
        source_path = chunk.metadata.get("source", "")
        if chunk.source_doc_id:
            source_path = chunk.source_doc_id

        # 获取 section_path（如果有标题或章节信息）
        section_path = ""
        if "title" in chunk.metadata:
            section_path = str(chunk.metadata["title"])
        elif "heading" in chunk.metadata:
            section_path = str(chunk.metadata["heading"])

        # 如果有明确的位置信息，也包含进来
        if chunk.chunk_index is not None:
            section_path += f"_idx{chunk.chunk_index}"

        # 计算 content_hash
        content_hash = self._hash_text(chunk.text)

        # 组合生成最终 chunk_id
        id_components = f"{source_path}|{section_path}|{content_hash}"
        chunk_id = self._hash_text(id_components)

        return chunk_id

    def _hash_text(self, text: str) -> str:
        """计算文本的 SHA256 哈希。

        Args:
            text: 输入文本。

        Returns:
            SHA256 哈希字符串（前16位）。
        """
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        # 使用前16位哈希，足够唯一且更紧凑
        return hash_obj.hexdigest()[:16]

    def upsert_single(
        self,
        chunk: Chunk,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        trace: Optional["TraceContext"] = None,
    ) -> str:
        """存储单个 chunk。

        Args:
            chunk: Chunk 对象。
            dense_vector: 稠密向量。
            sparse_vector: 稀疏向量。
            trace: 可选追踪上下文。

        Returns:
            生成的 chunk_id。
        """
        chunk_ids = self.upsert(
            [chunk],
            [dense_vector],
            [sparse_vector],
            trace=trace,
        )
        return chunk_ids[0]

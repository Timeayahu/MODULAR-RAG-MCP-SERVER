"""BM25Indexer 实现 - BM25 倒排索引存储。

根据 DEV_SPEC 3.1.1 Storage 阶段：
- 存储后端：持久化存储 Sparse Vector 到 data/db/bm25/
- 倒排索引：构建 term -> [chunk_ids] 的映射
- 支持查询：根据 query terms 返回相关 chunk_ids 及分数
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BM25Indexer:
    """BM25 倒排索引器。

    特性：
    - 构建 term -> chunk_ids 倒排索引
    - 持久化到磁盘（JSON 格式）
    - 支持增量更新
    - 查询返回 Top-K chunk_ids
    """

    def __init__(self, index_dir: str = "data/db/bm25") -> None:
        """初始化 BM25Indexer。

        Args:
            index_dir: 索引目录路径。
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 倒排索引：term -> {chunk_id: weight}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)

        # chunk_id -> chunk 信息（用于查询时返回）
        self.chunk_info: Dict[str, Dict] = {}

    def build(
        self,
        chunk_ids: List[str],
        sparse_vectors: List[Dict[str, float]],
        chunk_metadata: Optional[List[Dict]] = None,
    ) -> None:
        """构建 BM25 索引。

        Args:
            chunk_ids: Chunk ID 列表。
            sparse_vectors: 稀疏向量列表（{term: weight}）。
            chunk_metadata: 可选的 chunk 元数据列表。

        Raises:
            ValueError: 如果输入列表长度不一致。
        """
        if len(chunk_ids) != len(sparse_vectors):
            raise ValueError(
                f"chunk_ids 数量 ({len(chunk_ids)}) 与 "
                f"sparse_vectors 数量 ({len(sparse_vectors)}) 不一致"
            )

        if chunk_metadata is not None and len(chunk_ids) != len(chunk_metadata):
            raise ValueError(
                f"chunk_ids 数量 ({len(chunk_ids)}) 与 "
                f"chunk_metadata 数量 ({len(chunk_metadata)}) 不一致"
            )

        logger.info(f"开始构建 BM25 索引: {len(chunk_ids)} chunks")

        # 构建倒排索引
        for i, (chunk_id, sparse_vec) in enumerate(zip(chunk_ids, sparse_vectors)):
            # 添加到倒排索引
            for term, weight in sparse_vec.items():
                self.inverted_index[term][chunk_id] = weight

            # 存储 chunk 信息
            self.chunk_info[chunk_id] = {
                "index": i,
                "metadata": chunk_metadata[i] if chunk_metadata else {},
            }

        logger.info(
            f"BM25 索引构建完成: {len(self.inverted_index)} terms, "
            f"{len(self.chunk_info)} chunks"
        )

    def query(
        self,
        query_terms: Dict[str, float],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """查询 BM25 索引。

        Args:
            query_terms: 查询 terms 及其权重 {term: weight}。
            top_k: 返回 Top-K 结果。

        Returns:
            [(chunk_id, score)] 列表，按分数降序排列。
        """
        if not query_terms:
            return []

        # 累积每个 chunk 的分数
        chunk_scores: Dict[str, float] = defaultdict(float)

        for term, query_weight in query_terms.items():
            if term in self.inverted_index:
                # 获取包含该 term 的所有 chunks
                for chunk_id, doc_weight in self.inverted_index[term].items():
                    # 分数 = query_weight * doc_weight
                    chunk_scores[chunk_id] += query_weight * doc_weight

        # 排序并返回 Top-K
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:top_k]

    def save(self, collection_name: str = "default") -> None:
        """保存索引到磁盘。

        Args:
            collection_name: 集合名称（用于区分不同的索引）。
        """
        index_file = self.index_dir / f"{collection_name}_index.json"
        chunk_info_file = self.index_dir / f"{collection_name}_chunks.json"

        # 保存倒排索引
        with open(index_file, "w", encoding="utf-8") as f:
            # 转换为可序列化格式
            serializable_index = {
                term: dict(chunks) for term, chunks in self.inverted_index.items()
            }
            json.dump(serializable_index, f, ensure_ascii=False, indent=2)

        # 保存 chunk 信息
        with open(chunk_info_file, "w", encoding="utf-8") as f:
            json.dump(self.chunk_info, f, ensure_ascii=False, indent=2)

        logger.info(f"BM25 索引已保存到 {self.index_dir} (collection: {collection_name})")

    def load(self, collection_name: str = "default") -> None:
        """从磁盘加载索引。

        Args:
            collection_name: 集合名称。

        Raises:
            FileNotFoundError: 如果索引文件不存在。
        """
        index_file = self.index_dir / f"{collection_name}_index.json"
        chunk_info_file = self.index_dir / f"{collection_name}_chunks.json"

        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")

        # 加载倒排索引
        with open(index_file, "r", encoding="utf-8") as f:
            loaded_index = json.load(f)
            self.inverted_index = defaultdict(dict)
            for term, chunks in loaded_index.items():
                self.inverted_index[term] = chunks

        # 加载 chunk 信息
        if chunk_info_file.exists():
            with open(chunk_info_file, "r", encoding="utf-8") as f:
                self.chunk_info = json.load(f)

        logger.info(
            f"BM25 索引已加载: {len(self.inverted_index)} terms, "
            f"{len(self.chunk_info)} chunks (collection: {collection_name})"
        )

    def clear(self) -> None:
        """清空当前索引。"""
        self.inverted_index.clear()
        self.chunk_info.clear()
        logger.debug("BM25 索引已清空")

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小。"""
        return len(self.inverted_index)

    @property
    def num_chunks(self) -> int:
        """返回索引的 chunk 数量。"""
        return len(self.chunk_info)

    def get_chunk_info(self, chunk_id: str) -> Optional[Dict]:
        """获取 chunk 信息。

        Args:
            chunk_id: Chunk ID。

        Returns:
            Chunk 信息字典，如果不存在则返回 None。
        """
        return self.chunk_info.get(chunk_id)

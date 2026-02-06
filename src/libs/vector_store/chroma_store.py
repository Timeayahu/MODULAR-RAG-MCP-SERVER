"""ChromaStore 实现，基于 LlamaIndex ChromaVectorStore 适配器。

根据 DEV_SPEC 3.3.3：
- 本项目选用 Chroma 作为向量数据库
- LlamaIndex 提供了成熟的 ChromaVectorStore 适配器，与 Ingestion Pipeline 无缝集成
- Chroma 采用嵌入式设计，pip install chromadb 即可使用，无需额外部署数据库服务
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from libs.vector_store.base_vector_store import BaseVectorStore, LLAMA_INDEX_AVAILABLE

# Chroma 依赖
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext
    from llama_index.core.vector_stores import VectorStore as LlamaVectorStore


class ChromaStore(BaseVectorStore):
    """Chroma 向量数据库实现，基于 LlamaIndex ChromaVectorStore 适配器。

    特性：
    - 嵌入式设计，无需额外部署
    - 支持本地持久化
    - 与 LlamaIndex Ingestion Pipeline 无缝集成
    """

    provider_name: str = "chroma"

    def __init__(self, config) -> None:
        super().__init__(config)

        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb 未安装，请运行: pip install chromadb")

        # 创建 Chroma 客户端
        self._chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 获取或创建 collection
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.config.collection_name
        )

    def get_llama_vector_store(self) -> "LlamaVectorStore":
        """获取 LlamaIndex ChromaVectorStore 实例。"""
        if self._llama_vector_store is not None:
            return self._llama_vector_store

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError:
            raise ImportError(
                "llama-index-vector-stores-chroma 未安装，"
                "请运行: pip install llama-index-vector-stores-chroma"
            )

        # 使用已有的 chroma collection 创建 LlamaIndex 适配器
        self._llama_vector_store = ChromaVectorStore(
            chroma_collection=self._collection
        )

        return self._llama_vector_store

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional["TraceContext"] = None,
    ) -> None:
        """写入或更新向量记录。

        支持两种方式：
        1. 直接使用 Chroma API（当前方法）
        2. 通过 LlamaIndex VectorStoreIndex（get_llama_vector_store）

        Args:
            records: 记录列表，每条记录包含 id, vector, text, metadata。
            trace: 可选追踪上下文。
        """
        if not isinstance(records, list):
            raise ValueError("chroma records 格式错误")
        if not records:
            return

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for record in records:
            if not isinstance(record, dict):
                raise ValueError("chroma records 格式错误")
            record_id = record.get("id")
            vector = record.get("vector")
            if not isinstance(record_id, str) or not isinstance(vector, list):
                raise ValueError("chroma records 缺少 id/vector")

            ids.append(record_id)
            embeddings.append([float(value) for value in vector])
            documents.append(str(record.get("text", "")))

            # 处理 metadata，确保所有值都是 Chroma 支持的类型
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError("chroma metadata 格式错误")
            # Chroma 只支持 str, int, float, bool 类型的 metadata 值
            clean_metadata = self._clean_metadata(metadata)
            metadatas.append(clean_metadata)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, Any]]:
        """向量查询。

        Args:
            vector: 查询向量。
            top_k: 返回数量。
            filters: 可选过滤条件（Chroma where 语法）。
            trace: 可选追踪上下文。

        Returns:
            命中记录列表。
        """
        if not isinstance(vector, list):
            raise ValueError("chroma vector 格式错误")
        if top_k <= 0:
            raise ValueError("chroma top_k 必须大于 0")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("chroma filters 格式错误")

        results = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        output: List[Dict[str, Any]] = []
        for idx, record_id in enumerate(ids):
            distance = dists[idx] if idx < len(dists) else None
            # Chroma 返回的是 L2 距离，转换为相似度分数
            # 距离越小越相似，分数越高越相似，取负数让距离小的分数高
            score = -float(distance) if distance is not None else 0.0
            output.append(
                {
                    "id": record_id,
                    "score": score,
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                }
            )
        return output

    def delete(
        self,
        ids: List[str],
        trace: Optional["TraceContext"] = None,
    ) -> None:
        """删除指定记录。

        Args:
            ids: 要删除的记录 ID 列表。
            trace: 可选追踪上下文。
        """
        if not ids:
            return
        self._collection.delete(ids=ids)

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理 metadata，确保所有值都是 Chroma 支持的类型。

        Chroma 只支持 str, int, float, bool 类型的 metadata 值。
        列表和嵌套字典会被转换为 JSON 字符串。
        """
        import json

        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                # 转换为 JSON 字符串
                clean[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                # 跳过 None 值
                continue
            else:
                # 其他类型转换为字符串
                clean[key] = str(value)
        return clean

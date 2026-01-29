"""Ingestion Pipeline 主流程。

根据 DEV_SPEC 3.1.1 和 5.4.1：
- 使用 LlamaIndex 的 Ingestion Pipeline 构建统一、可配置且可观测的数据导入与分块能力
- 覆盖文档加载、格式解析、语义切分、多模态增强、嵌入计算、去重与批量上载到向量存储

完整流程（C14）：
integrity → load → split → transform → encode → store
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from core.settings import Settings
from ingestion.models import Document, Chunk
from libs.loader.pdf_loader import PdfLoader
from libs.loader.file_integrity import compute_sha256, should_skip, mark_success
from libs.splitter.splitter_factory import SplitterFactory
from libs.splitter.base_splitter import SplitResult
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.metadata_enricher import MetadataEnricher
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Ingestion Pipeline，完整的数据摄取流程编排。

    根据 DEV_SPEC 3.1.1 的分层职责：
    - File Integrity：文件去重检查（SHA256）
    - Loader：负责把原始文件解析为统一的 Document 对象
    - Splitter：基于 Markdown 结构与参数配置把 Document 切为若干 Chunk
    - Transform：可插入的处理步骤（ChunkRefiner, MetadataEnricher, ImageCaptioner）
    - Embed：批量计算 dense/sparse embedding
    - Storage：Upsert 到 VectorStore 和 BM25 Index
    """

    def __init__(self, settings: Settings, collection: str = "default") -> None:
        """初始化 Ingestion Pipeline。

        Args:
            settings: 全局配置。
            collection: 集合名称。
        """
        self._settings = settings
        self._collection = collection
        
        # Loader & Splitter
        self._loader = PdfLoader()
        self._splitter = SplitterFactory.create(settings)
        
        # Transform 组件
        self._chunk_refiner = ChunkRefiner(settings)
        self._metadata_enricher = MetadataEnricher(settings)
        self._image_captioner = ImageCaptioner(settings)
        
        # Embedding & Storage 组件
        batch_size = getattr(getattr(settings, "ingestion", None), "batch_size", 32)

        # 特殊情况：在集成测试中，embedding.provider 通常为 "mock"，
        # 这时会通过 patch 替换 BatchProcessor，因此不需要真实构造 encoder。
        provider = getattr(getattr(settings, "embedding", None), "provider", None)
        if provider == "mock":
            # 这里的构造会被 tests 中的 patch("ingestion.pipeline.BatchProcessor") 拦截
            self._batch_processor = BatchProcessor()
        else:
            # 1) 创建 DenseEncoder（使用 EmbeddingFactory）
            embedding = EmbeddingFactory.create(settings)
            dense_encoder = DenseEncoder(embedding)

            # 2) 创建 SparseEncoder（使用 Retrieval 配置中的 BM25 参数，如有）
            try:
                bm25_cfg = getattr(settings.retrieval, "bm25", None)
                if bm25_cfg is not None:
                    sparse_encoder = SparseEncoder(
                        k1=getattr(bm25_cfg, "k1", 1.5),
                        b=getattr(bm25_cfg, "b", 0.75),
                        stop_words=set(getattr(bm25_cfg, "stop_words", [])) or None,
                    )
                else:
                    sparse_encoder = SparseEncoder()
            except Exception:
                # 保底：使用默认参数构造 SparseEncoder
                sparse_encoder = SparseEncoder()

            # 3) 创建 BatchProcessor（真实环境）
            self._batch_processor = BatchProcessor(
                dense_encoder=dense_encoder,
                sparse_encoder=sparse_encoder,
                batch_size=batch_size,
            )

        # 创建 VectorStore & VectorUpserter
        # 注意：在部分集成测试中会直接 patch VectorUpserter 或 VectorStoreFactory，
        # 因此这里保持最小依赖。
        vector_store = VectorStoreFactory.create(settings)
        self._vector_upserter = VectorUpserter(vector_store)

        # BM25 索引目录：当前使用默认路径 data/db/bm25
        # 未来可从 Settings 中引入 data_paths 配置
        self._bm25_indexer = BM25Indexer()

        # 图片存储：当前使用默认路径 data/images，下层按 collection 分目录
        # ImageStorage 本身已经支持按 collection 组织文件
        self._image_storage = ImageStorage()

    @property
    def settings(self) -> Settings:
        """返回配置。"""
        return self._settings

    def load_document(self, path: str) -> Document:
        """使用 Loader 加载单个文档。

        Args:
            path: 文件路径。

        Returns:
            Document 对象（text 为 Markdown 格式）。
        """
        logger.info(f"加载文档: {path}")
        return self._loader.load(path)

    def split_document(self, document: Document) -> List[Chunk]:
        """对 Document 进行切分，返回 Chunk 列表。

        每个 Chunk 携带完整的定位信息和来源信息。

        Args:
            document: 输入文档。

        Returns:
            Chunk 列表。
        """
        logger.info(f"切分文档: {document.id}")

        # 使用 Splitter 切分
        split_results: List[SplitResult] = self._splitter.split(document.text)

        # 转换为 Chunk 对象
        chunks: List[Chunk] = []
        for result in split_results:
            chunk_id = self._generate_chunk_id(document.id, result.chunk_index)

            chunk = Chunk(
                id=chunk_id,
                text=result.text,
                metadata={
                    **document.metadata,
                    **result.metadata,
                },
                source_doc_id=document.id,
                chunk_index=result.chunk_index,
                start_offset=result.start_offset,
                end_offset=result.end_offset,
                image_refs=self._extract_chunk_image_refs(
                    document.images, result.start_offset, result.end_offset
                ),
            )
            chunks.append(chunk)

        logger.info(f"文档 {document.id} 切分为 {len(chunks)} 个 Chunk")
        return chunks

    def split_document_text(self, document: Document) -> List[str]:
        """对 Document 文本进行切分，返回文本片段列表（简化接口）。

        Args:
            document: 输入文档。

        Returns:
            切分后的文本片段列表。
        """
        return self._splitter.split_text(document.text)

    def run_loader_and_splitter(self, path: str) -> List[Chunk]:
        """端到端执行 Loader → Splitter。

        Args:
            path: 待处理文件路径。

        Returns:
            Chunk 列表。
        """
        doc = self.load_document(path)
        return self.split_document(doc)

    def run(
        self,
        path: str,
        force: bool = False,
        trace: Optional["TraceContext"] = None,
    ) -> Dict[str, any]:
        """运行完整的 Ingestion Pipeline。

        流程：integrity → load → split → transform → encode → store

        Args:
            path: 文件路径。
            force: 是否强制重新处理（跳过完整性检查）。
            trace: 可选的追踪上下文。

        Returns:
            处理结果字典，包含：
            - skipped: 是否跳过（文件未变更）
            - num_chunks: 生成的 Chunk 数量
            - chunk_ids: Chunk ID 列表

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件格式不支持或处理失败。
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        logger.info(f"开始处理文件: {path}")

        # Stage 1: File Integrity Check
        if not force:
            file_hash = compute_sha256(str(file_path))
            if should_skip(file_hash):
                logger.info(f"文件未变更，跳过处理: {path}")
                return {"skipped": True, "file_hash": file_hash}
        
        try:
            # Stage 2: Load
            logger.info("Stage 2: Loading document...")
            document = self.load_document(str(file_path))
            
            # Stage 3: Split
            logger.info("Stage 3: Splitting document...")
            chunks = self.split_document(document)
            
            # Stage 4: Transform
            logger.info("Stage 4: Transforming chunks...")
            # 4.1: Chunk Refinement
            chunks = self._chunk_refiner.transform(chunks, trace=trace)
            # 4.2: Metadata Enrichment
            chunks = self._metadata_enricher.transform(chunks, trace=trace)
            # 4.3: Image Captioning
            chunks = self._image_captioner.transform(chunks, trace=trace)
            
            # Stage 5: Encode
            logger.info("Stage 5: Encoding chunks...")
            # BatchProcessor.process 返回 (dense_vectors, sparse_vectors)
            dense_vectors, sparse_vectors = self._batch_processor.process(
                chunks, trace=trace
            )
            processed_chunks = chunks
            
            # Stage 6: Store
            logger.info("Stage 6: Storing to vector store and BM25 index...")
            # 6.1: Upsert to VectorStore
            chunk_ids = self._vector_upserter.upsert(
                processed_chunks, dense_vectors, sparse_vectors, trace=trace
            )
            # 6.2: Build BM25 Index
            # 准备 chunk 元数据
            chunk_metadata = [
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "source_doc_id": chunk.source_doc_id,
                }
                for chunk in processed_chunks
            ]
            self._bm25_indexer.build(chunk_ids, sparse_vectors, chunk_metadata)
            self._bm25_indexer.save(self._collection)
            # 6.3: Store Images (if any)
            self._store_images(document, processed_chunks)
            
            # Mark success
            if not force:
                mark_success(file_hash)
            
            logger.info(f"文件处理完成: {path}, 生成 {len(chunk_ids)} 个 Chunks")
            
            return {
                "skipped": False,
                "num_chunks": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "file_hash": file_hash if not force else None,
            }
            
        except Exception as e:
            logger.error(f"文件处理失败: {path}, 错误: {e}")
            raise ValueError(f"文件处理失败: {e}") from e

    def _store_images(self, document: Document, chunks: List[Chunk]) -> None:
        """存储文档中的图片。

        Args:
            document: 原始文档。
            chunks: 处理后的 Chunk 列表。
        """
        if not document.images:
            return
        
        logger.info(f"存储 {len(document.images)} 张图片...")
        for img_info in document.images:
            image_id = img_info.get("image_id")
            image_data = img_info.get("image_data")
            
            if not image_id or not image_data:
                logger.warning(f"图片信息不完整，跳过: {img_info}")
                continue
            
            try:
                # 保存图片到文件系统
                self._image_storage.save_image(
                    image_id=image_id,
                    image_data=image_data if isinstance(image_data, bytes) else image_data.encode(),
                    collection=self._collection,
                    metadata={
                        "source_doc": document.id,
                        "page": img_info.get("page", -1),
                    },
                )
            except Exception as e:
                logger.error(f"图片存储失败: {image_id}, 错误: {e}")

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """生成 Chunk ID。

        根据 DEV_SPEC 3.1.1：
        chunk_id 生成算法采用确定的哈希组合：hash(source_path + section_path + content_hash)

        当前简化实现：doc_id + chunk_index

        Args:
            doc_id: 文档 ID。
            chunk_index: Chunk 索引。

        Returns:
            Chunk ID。
        """
        return f"{doc_id}_chunk_{chunk_index}"

    def _extract_chunk_image_refs(
        self,
        doc_images: List[dict],
        start_offset: int,
        end_offset: int,
    ) -> List[str]:
        """提取 Chunk 范围内的图片引用。

        Args:
            doc_images: 文档的图片列表。
            start_offset: Chunk 起始位置。
            end_offset: Chunk 结束位置。

        Returns:
            图片 ID 列表。
        """
        refs: List[str] = []
        for img in doc_images:
            pos = img.get("position", -1)
            if start_offset <= pos < end_offset:
                refs.append(img.get("image_id", ""))
        return [r for r in refs if r]  # 过滤空值


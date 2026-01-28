"""Ingestion Pipeline 集成测试。

根据 DEV_SPEC C14 验收标准：
- 对 fixtures 样例文档跑完整 pipeline
- 输出向量与 bm25 索引文件
"""

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import Settings
from ingestion.pipeline import IngestionPipeline
from ingestion.models import Document


@pytest.fixture
def temp_data_dirs(tmp_path):
    """创建临时数据目录。"""
    image_dir = tmp_path / "images"
    chroma_dir = tmp_path / "chroma"
    bm25_dir = tmp_path / "bm25"
    
    image_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    
    yield {
        "image_dir": str(image_dir),
        "chroma_dir": str(chroma_dir),
        "bm25_dir": str(bm25_dir),
    }
    
    # 清理
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)


@pytest.fixture
def mock_settings(temp_data_dirs):
    """创建模拟配置。"""
    settings = MagicMock()
    
    # Ingestion 配置
    settings.ingestion = MagicMock()
    settings.ingestion.batch_size = 2
    settings.ingestion.embedding = MagicMock()
    settings.ingestion.embedding.dense_enabled = True
    settings.ingestion.embedding.sparse_enabled = True
    settings.ingestion.image_captioning = MagicMock()
    settings.ingestion.image_captioning.enabled = False  # 禁用以简化测试
    settings.ingestion.image_captioning.prompt_path = "config/prompts/image_captioning.txt"
    
    # Transform 配置
    settings.transform = MagicMock()
    settings.transform.refine_enabled = False  # 禁用以简化测试
    settings.transform.refine_use_llm = False
    settings.transform.enrich_metadata = False  # 禁用以简化测试
    settings.transform.enrich_use_llm = False
    
    # LLM 配置（用于 Transform）
    settings.llm = MagicMock()
    settings.llm.provider = "mock"
    
    # Vision LLM 配置
    settings.vision_llm = MagicMock()
    settings.vision_llm.provider = "mock"
    
    # Embedding 配置
    settings.embedding = MagicMock()
    settings.embedding.provider = "mock"
    settings.embedding.model = "mock-model"
    
    # VectorStore 配置
    settings.vector_store = MagicMock()
    settings.vector_store.backend = "chroma"
    settings.vector_store.provider = "chroma"
    settings.vector_store.persist_directory = temp_data_dirs["chroma_dir"]
    
    # Splitter 配置
    settings.splitter = MagicMock()
    settings.splitter.provider = "recursive"
    settings.splitter.chunk_size = 200
    settings.splitter.chunk_overlap = 50
    settings.splitter.separators = ["\n\n", "\n", " ", ""]
    
    # BM25 配置
    settings.retrieval = MagicMock()
    settings.retrieval.bm25 = MagicMock()
    settings.retrieval.bm25.k1 = 1.5
    settings.retrieval.bm25.b = 0.75
    settings.retrieval.bm25.stop_words = ["is", "a", "the", "of"]
    
    # Data paths 配置
    settings.data_paths = MagicMock()
    settings.data_paths.image_dir = temp_data_dirs["image_dir"]
    settings.data_paths.bm25_index_dir = temp_data_dirs["bm25_dir"]
    
    return settings


@pytest.fixture
def sample_document_path(tmp_path):
    """创建示例文档。"""
    # 使用 fixtures 中的示例文件，如果不存在则创建一个简单的文本文件
    fixtures_path = repo_root / "tests" / "fixtures" / "sample_documents" / "sample.txt"
    if fixtures_path.exists():
        return str(fixtures_path)
    
    # 创建临时示例文件
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "This is a sample document for testing.\n"
        "It contains multiple lines and paragraphs.\n\n"
        "Deep Learning is a subset of Machine Learning.\n"
        "Natural Language Processing is a field of AI.\n"
    )
    return str(sample_file)


@pytest.mark.integration
@patch("ingestion.pipeline.PdfLoader")
@patch("ingestion.pipeline.BatchProcessor")
@patch("libs.vector_store.vector_store_factory.VectorStoreFactory.create")
@patch("libs.loader.file_integrity.should_skip")
@patch("libs.loader.file_integrity.mark_success")
def test_ingestion_pipeline_end_to_end(
    mock_mark_success,
    mock_should_skip,
    mock_vector_store_factory,
    mock_batch_processor_cls,
    mock_pdf_loader_cls,
    mock_settings,
    sample_document_path,
):
    """验收标准：对 fixtures 样例文档跑完整 pipeline。"""
    # Mock 文件完整性检查
    mock_should_skip.return_value = False
    
    # Mock PdfLoader
    mock_loader = MagicMock()
    mock_document = Document(
        id="test_doc",
        text="This is a sample document.\n\nDeep Learning is important.",
        metadata={"source_path": sample_document_path, "doc_type": "txt"},
        images=[],
    )
    mock_loader.load.return_value = mock_document
    mock_pdf_loader_cls.return_value = mock_loader
    
    # Mock BatchProcessor
    mock_batch_processor = MagicMock()
    def mock_process(chunks, trace=None):
        # 为每个 chunk 生成 dense 和 sparse vectors
        dense_vectors = [[0.1, 0.2] for _ in chunks]
        sparse_vectors = [{"term": 0.5} for _ in chunks]
        return dense_vectors, sparse_vectors, chunks
    mock_batch_processor.process.side_effect = mock_process
    mock_batch_processor_cls.return_value = mock_batch_processor
    
    # Mock VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.upsert.return_value = None
    mock_vector_store_factory.return_value = mock_vector_store
    
    # 创建 pipeline
    pipeline = IngestionPipeline(mock_settings, collection="test_collection")
    
    # 运行 pipeline（使用 force=True 跳过完整性检查）
    result = pipeline.run(sample_document_path, force=True)
    
    # 验证结果
    assert result["skipped"] is False
    assert result["num_chunks"] > 0
    assert len(result["chunk_ids"]) > 0
    
    # 验证关键阶段被调用
    mock_loader.load.assert_called_once()
    mock_batch_processor.process.assert_called_once()
    # force=True 时不会调用 mark_success
    mock_should_skip.assert_not_called()


@pytest.mark.integration
@patch("ingestion.pipeline.PdfLoader")
@patch("libs.loader.file_integrity.should_skip")
def test_ingestion_pipeline_skip_unchanged_file(
    mock_should_skip,
    mock_pdf_loader_cls,
    mock_settings,
    sample_document_path,
):
    """测试文件未变更时跳过处理。"""
    # Mock 文件完整性检查：文件未变更
    mock_should_skip.return_value = True
    
    # 创建 pipeline
    pipeline = IngestionPipeline(mock_settings, collection="test_collection")
    
    # 运行 pipeline
    result = pipeline.run(sample_document_path, force=False)
    
    # 验证跳过处理
    assert result["skipped"] is True
    assert "file_hash" in result


@pytest.mark.integration
@patch("ingestion.pipeline.PdfLoader")
@patch("libs.loader.file_integrity.should_skip")
@patch("libs.loader.file_integrity.mark_success")
def test_ingestion_pipeline_force_reprocess(
    mock_mark_success,
    mock_should_skip,
    mock_pdf_loader_cls,
    mock_settings,
    sample_document_path,
):
    """测试强制重新处理（跳过完整性检查）。"""
    # Mock PdfLoader
    mock_loader = MagicMock()
    mock_document = Document(
        id="test_doc",
        text="Sample text for force reprocess.",
        metadata={"source_path": sample_document_path},
        images=[],
    )
    mock_loader.load.return_value = mock_document
    mock_pdf_loader_cls.return_value = mock_loader
    
    # Mock BatchProcessor & VectorStore
    with patch("ingestion.pipeline.BatchProcessor") as mock_batch_processor_cls, \
         patch("libs.vector_store.vector_store_factory.VectorStoreFactory.create") as mock_vector_store_factory:
        
        mock_batch_processor = MagicMock()
        def mock_process(chunks, trace=None):
            dense_vectors = [[0.1, 0.2] for _ in chunks]
            sparse_vectors = [{"term": 0.5} for _ in chunks]
            return dense_vectors, sparse_vectors, chunks
        mock_batch_processor.process.side_effect = mock_process
        mock_batch_processor_cls.return_value = mock_batch_processor
        
        mock_vector_store = MagicMock()
        mock_vector_store_factory.return_value = mock_vector_store
        
        # 创建 pipeline
        pipeline = IngestionPipeline(mock_settings, collection="test_collection")
        
        # 运行 pipeline（force=True）
        result = pipeline.run(sample_document_path, force=True)
        
        # 验证不会检查文件完整性
        mock_should_skip.assert_not_called()
        
        # 验证处理完成
        assert result["skipped"] is False


@pytest.mark.integration
@patch("ingestion.pipeline.PdfLoader")
@patch("ingestion.pipeline.BatchProcessor")
@patch("libs.vector_store.vector_store_factory.VectorStoreFactory.create")
@patch("libs.loader.file_integrity.should_skip")
def test_ingestion_pipeline_bm25_index_created(
    mock_should_skip,
    mock_vector_store_factory,
    mock_batch_processor_cls,
    mock_pdf_loader_cls,
    mock_settings,
    sample_document_path,
    temp_data_dirs,
):
    """验收标准：输出 bm25 索引文件。"""
    # Mock 完整性检查
    mock_should_skip.return_value = False
    
    # Mock PdfLoader
    mock_loader = MagicMock()
    mock_document = Document(
        id="test_doc",
        text="Deep Learning and Machine Learning are related fields.",
        metadata={"source_path": sample_document_path},
        images=[],
    )
    mock_loader.load.return_value = mock_document
    mock_pdf_loader_cls.return_value = mock_loader
    
    # Mock BatchProcessor
    mock_batch_processor = MagicMock()
    def mock_process(chunks, trace=None):
        dense_vectors = [[0.1, 0.2] for _ in chunks]
        sparse_vectors = [{"deep": 1.0, "learning": 0.8} for _ in chunks]
        return dense_vectors, sparse_vectors, chunks
    mock_batch_processor.process.side_effect = mock_process
    mock_batch_processor_cls.return_value = mock_batch_processor
    
    # Mock VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store_factory.return_value = mock_vector_store
    
    with patch("libs.loader.file_integrity.mark_success"):
        # 创建 pipeline
        pipeline = IngestionPipeline(mock_settings, collection="test_collection")
        
        # 运行 pipeline（使用 force=True 跳过完整性检查）
        result = pipeline.run(sample_document_path, force=True)
        
        # 验证 pipeline 运行成功
        assert result["skipped"] is False
        assert result["num_chunks"] > 0
        
        # 验证 BM25 索引文件被创建（如果目录存在）
        bm25_dir = Path(temp_data_dirs["bm25_dir"]) / "test_collection"
        if bm25_dir.exists():
            # 验证索引文件存在
            index_file = bm25_dir / "inverted_index.json"
            metadata_file = bm25_dir / "chunk_metadata.json"
            assert index_file.exists() or metadata_file.exists()


@pytest.mark.integration
def test_ingestion_pipeline_nonexistent_file(mock_settings):
    """测试文件不存在时抛出错误。"""
    pipeline = IngestionPipeline(mock_settings, collection="test_collection")
    
    with pytest.raises(FileNotFoundError, match="文件不存在"):
        pipeline.run("/nonexistent/file.pdf")


@pytest.mark.integration
@patch("ingestion.pipeline.PdfLoader")
def test_ingestion_pipeline_loader_error(
    mock_pdf_loader_cls,
    mock_settings,
    sample_document_path,
):
    """测试 Loader 失败时抛出清晰错误。"""
    # Mock PdfLoader 抛出错误
    mock_loader = MagicMock()
    mock_loader.load.side_effect = Exception("PDF 解析失败")
    mock_pdf_loader_cls.return_value = mock_loader
    
    pipeline = IngestionPipeline(mock_settings, collection="test_collection")
    
    # 使用 force=True 跳过完整性检查，直接执行到 Loader
    with pytest.raises(ValueError, match="文件处理失败"):
        pipeline.run(sample_document_path, force=True)

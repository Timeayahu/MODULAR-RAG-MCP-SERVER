"""MetadataEnricher 单元测试。"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import (
    EmbeddingConfig,
    EvaluationConfig,
    LLMConfig,
    ObservabilityConfig,
    RetrievalConfig,
    RerankConfig,
    Settings,
    SplitterConfig,
    TransformConfig,
    VectorStoreConfig,
)
from ingestion.models import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher


def _build_test_settings(enrich_use_llm: bool = False) -> Settings:
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterConfig(provider="recursive"),
        transform=TransformConfig(enrich_metadata=True, enrich_use_llm=enrich_use_llm),
        vector_store=VectorStoreConfig(provider="chroma"),
        retrieval=RetrievalConfig(),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(),
        observability=ObservabilityConfig(),
    )


@pytest.mark.unit
def test_rule_based_enrichment():
    settings = _build_test_settings(enrich_use_llm=False)
    enricher = MetadataEnricher(settings)
    
    text = "# Project Overview\nThis is a sample document for RAG system."
    chunks = [Chunk(id="1", text=text)]
    
    enriched = enricher.transform(chunks)
    
    assert enriched[0].metadata["title"] == "Project Overview"
    assert "sample document" in enriched[0].metadata["summary"]
    assert "tags" in enriched[0].metadata


@pytest.mark.unit
def test_llm_based_enrichment_success():
    settings = _build_test_settings(enrich_use_llm=True)
    
    mock_llm = MagicMock()
    mock_data = {
        "title": "LLM Title",
        "summary": "LLM Summary",
        "tags": ["AI", "RAG"]
    }
    mock_llm.chat.return_value = json.dumps(mock_data)
    
    with patch("libs.llm.llm_factory.LLMFactory.create", return_value=mock_llm):
        enricher = MetadataEnricher(settings)
        chunks = [Chunk(id="1", text="Sample Text")]
        enriched = enricher.transform(chunks)
        
        assert enriched[0].metadata["title"] == "LLM Title"
        assert enriched[0].metadata["summary"] == "LLM Summary"
        assert "AI" in enriched[0].metadata["tags"]


@pytest.mark.unit
def test_llm_based_enrichment_fallback_on_error():
    settings = _build_test_settings(enrich_use_llm=True)
    
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = Exception("LLM Error")
    
    with patch("libs.llm.llm_factory.LLMFactory.create", return_value=mock_llm):
        enricher = MetadataEnricher(settings)
        chunks = [Chunk(id="1", text="# Rule Title\nContent")]
        enriched = enricher.transform(chunks)
        
        # 应保留规则模式结果并标记 fallback
        assert enriched[0].metadata["title"] == "Rule Title"
        assert enriched[0].metadata["enrich_fallback"] is True

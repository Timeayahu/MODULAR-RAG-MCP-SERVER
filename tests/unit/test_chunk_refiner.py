"""ChunkRefiner 单元测试。"""

import sys
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
from ingestion.transform.chunk_refiner import ChunkRefiner


def _build_test_settings(use_llm: bool = False) -> Settings:
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterConfig(provider="recursive"),
        transform=TransformConfig(refine_enabled=True, refine_use_llm=use_llm),
        vector_store=VectorStoreConfig(provider="chroma"),
        retrieval=RetrievalConfig(),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(),
        observability=ObservabilityConfig(),
    )


@pytest.mark.unit
def test_rule_based_cleanup():
    settings = _build_test_settings(use_llm=False)
    refiner = ChunkRefiner(settings)
    
    text = "  Hello \n\n\n World  \n Page 1 of 10 "
    chunks = [Chunk(id="1", text=text)]
    
    refined = refiner.transform(chunks)
    
    assert refined[0].text == "Hello\n\nWorld"


@pytest.mark.unit
def test_llm_refinement_success():
    settings = _build_test_settings(use_llm=True)
    
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "Refined Text"
    
    with patch("libs.llm.llm_factory.LLMFactory.create", return_value=mock_llm):
        refiner = ChunkRefiner(settings)
        chunks = [Chunk(id="1", text="Original Text")]
        refined = refiner.transform(chunks)
        
        assert refined[0].text == "Refined Text"
        mock_llm.chat.assert_called_once()


@pytest.mark.unit
def test_llm_refinement_fallback_on_error():
    settings = _build_test_settings(use_llm=True)
    
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = Exception("LLM Error")
    
    with patch("libs.llm.llm_factory.LLMFactory.create", return_value=mock_llm):
        refiner = ChunkRefiner(settings)
        chunks = [Chunk(id="1", text="  Original Text  \n Page 1 of 10 ")]
        refined = refiner.transform(chunks)
        
        # 应该降级到规则处理结果
        assert refined[0].text == "Original Text"
        assert refined[0].metadata.get("refine_fallback") is True

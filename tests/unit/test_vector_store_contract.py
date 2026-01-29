"""VectorStore 契约测试。"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """测试用 Fake VectorStore。"""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._records: List[Dict[str, Any]] = []

    def upsert(self, records, trace=None) -> None:  # type: ignore[override]
        self._records.extend(records)

    def query(self, vector, top_k, filters=None, trace=None):  # type: ignore[override]
        results = []
        for record in self._records[:top_k]:
            results.append(
                {
                    "id": record["id"],
                    "score": 1.0,
                    "text": record["text"],
                    "metadata": record.get("metadata", {}),
                }
            )
        return results


@pytest.mark.unit
def test_vector_store_contract_shapes():
    """契约测试：输入输出 shape 符合约定。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    store = FakeVectorStore(settings.vector_store)

    records = [
        {
            "id": "doc-1",
            "vector": [0.1, 0.2, 0.3],
            "text": "hello",
            "metadata": {"source": "unit"},
        },
        {
            "id": "doc-2",
            "vector": [0.2, 0.3, 0.4],
            "text": "world",
            "metadata": {"source": "unit"},
        },
    ]

    store.upsert(records)
    results = store.query(vector=[0.1, 0.2, 0.3], top_k=1, filters=None)

    assert len(results) == 1
    result = results[0]
    assert set(result.keys()) >= {"id", "score", "text", "metadata"}
    assert isinstance(result["id"], str)
    assert isinstance(result["score"], float)
    assert isinstance(result["text"], str)
    assert isinstance(result["metadata"], dict)


@pytest.mark.unit
def test_vector_store_factory_routes_provider():
    """工厂应能根据 provider 创建对应实现。"""
    try:
        VectorStoreFactory.register("fake", FakeVectorStore)
        settings = load_settings(str(repo_root / "config" / "settings.yaml"))
        settings.vector_store.provider = "fake"

        store = VectorStoreFactory.create(settings)
        assert isinstance(store, FakeVectorStore)
    finally:
        VectorStoreFactory.unregister("fake")

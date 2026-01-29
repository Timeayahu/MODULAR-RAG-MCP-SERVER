"""ChromaStore roundtrip 测试（可选）。"""

from pathlib import Path
import sys

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings


def _skip_if_missing_chroma() -> None:
    try:
        import chromadb  # noqa: F401
    except Exception:
        pytest.skip("chromadb 未安装，跳过 integration 测试")


@pytest.mark.integration
def test_chroma_store_roundtrip(tmp_path: Path):
    _skip_if_missing_chroma()

    from libs.vector_store.chroma_store import ChromaStore

    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.vector_store.provider = "chroma"
    settings.vector_store.persist_directory = str(tmp_path / "chroma")
    settings.vector_store.collection_name = "test_collection"

    store = ChromaStore(settings.vector_store)
    store.upsert(
        [
            {
                "id": "doc-1",
                "vector": [0.1, 0.2, 0.3],
                "text": "hello",
                "metadata": {"source": "test"},
            }
        ]
    )

    results = store.query(vector=[0.1, 0.2, 0.3], top_k=1, filters=None)
    assert results
    assert results[0]["id"] == "doc-1"

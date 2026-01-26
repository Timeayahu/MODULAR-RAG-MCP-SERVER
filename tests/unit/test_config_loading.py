"""配置加载与校验测试"""

import sys
from pathlib import Path
import pytest
import yaml

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings


@pytest.mark.unit
def test_load_settings_success():
    """加载默认配置应成功"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    assert settings.llm.provider
    assert settings.embedding.provider


@pytest.mark.unit
def test_load_settings_missing_field(tmp_path: Path):
    """缺少必填字段应报错并包含字段路径"""
    config_data = {
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "embedding": {"model": "text-embedding-3-small"},
        "vector_store": {"provider": "chroma"},
        "retrieval": {},
        "rerank": {},
        "evaluation": {},
        "observability": {},
    }

    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_settings(str(config_file))

    assert "embedding.provider" in str(excinfo.value)

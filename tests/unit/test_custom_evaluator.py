"""Custom Evaluator 测试。"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from core.settings import load_settings
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


@pytest.mark.unit
def test_custom_evaluator_metrics():
    """自定义评估器应输出稳定指标。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    evaluator = CustomEvaluator(settings.evaluation)

    metrics = evaluator.evaluate(
        query="q",
        retrieved_ids=["a", "b", "c"],
        golden_ids=["b"],
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0 / 2


@pytest.mark.unit
def test_evaluator_factory_routes_backend():
    """工厂应能根据 backend 创建对应实现。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.evaluation.backend = "custom"

    evaluator = EvaluatorFactory.create(settings)
    assert isinstance(evaluator, CustomEvaluator)


@pytest.mark.unit
def test_evaluator_factory_unknown_backend():
    """未知 backend 应报错。"""
    settings = load_settings(str(repo_root / "config" / "settings.yaml"))
    settings.evaluation.backend = "unknown"

    with pytest.raises(ValueError) as excinfo:
        EvaluatorFactory.create(settings)

    assert "unknown" in str(excinfo.value)

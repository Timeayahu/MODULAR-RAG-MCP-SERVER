"""
配置加载与校验模块

负责从 config/settings.yaml 加载配置并进行校验。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: int = 1536
    batch_size: int = 100


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    provider: str
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "knowledge_base"
    persist_directory: str = "data/db/chroma"


@dataclass
class RetrievalConfig:
    """检索配置"""
    dense: Dict[str, Any] = field(default_factory=lambda: {
        "top_k": 20,
        "score_threshold": 0.7
    })
    sparse: Dict[str, Any] = field(default_factory=lambda: {
        "top_k": 20,
        "score_threshold": 0.5
    })
    fusion: Dict[str, Any] = field(default_factory=lambda: {
        "method": "rrf",
        "rrf_k": 60
    })
    final_top_k: int = 10


@dataclass
class RerankConfig:
    """重排序配置"""
    enabled: bool = True
    backend: str = "none"
    model: Optional[str] = None
    top_k: int = 5
    score_threshold: float = 0.8


@dataclass
class EvaluationConfig:
    """评估配置"""
    enabled: bool = False
    backend: str = "custom"
    metrics: list = field(default_factory=lambda: ["hit_rate", "mrr", "ndcg"])
    golden_set_path: str = "tests/fixtures/golden_test_set.json"


@dataclass
class ObservabilityConfig:
    """可观测性配置"""
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "json",
        "file": "logs/app.log"
    })
    tracing: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "file": "logs/traces.jsonl"
    })
    dashboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "host": "localhost",
        "port": 8501
    })


@dataclass
class Settings:
    """主配置类"""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig


def _expand_env_vars(data: Any) -> Any:
    """递归展开环境变量"""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.getenv(env_var)
    else:
        return data


def load_settings(config_path: str) -> Settings:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Settings: 配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误或必填字段缺失
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    
    # 展开环境变量
    config_data = _expand_env_vars(raw_config)
    
    try:
        # 构建配置对象
        settings = Settings(
            llm=LLMConfig(**config_data['llm']),
            embedding=EmbeddingConfig(**config_data['embedding']),
            vector_store=VectorStoreConfig(**config_data['vector_store']),
            retrieval=RetrievalConfig(**config_data['retrieval']),
            rerank=RerankConfig(**config_data['rerank']),
            evaluation=EvaluationConfig(**config_data['evaluation']),
            observability=ObservabilityConfig(**config_data['observability'])
        )
        
        # 校验配置
        validate_settings(settings)
        
        return settings
        
    except (KeyError, TypeError) as e:
        raise ValueError(f"配置字段错误: {e}")


def validate_settings(settings: Settings) -> None:
    """
    校验配置的必填字段和合法性
    
    Args:
        settings: 配置对象
        
    Raises:
        ValueError: 配置校验失败
    """
    # 校验 LLM 配置
    if not settings.llm.provider:
        raise ValueError("llm.provider 不能为空")
    if not settings.llm.model:
        raise ValueError("llm.model 不能为空")
    
    # 校验 Embedding 配置
    if not settings.embedding.provider:
        raise ValueError("embedding.provider 不能为空")
    if not settings.embedding.model:
        raise ValueError("embedding.model 不能为空")
    
    # 校验 VectorStore 配置
    if not settings.vector_store.provider:
        raise ValueError("vector_store.provider 不能为空")
    
    # 校验数值范围
    if settings.retrieval.final_top_k <= 0:
        raise ValueError("retrieval.final_top_k 必须大于 0")
    
    if settings.rerank.top_k <= 0:
        raise ValueError("rerank.top_k 必须大于 0")
"""Local Embedding 实现，基于 LlamaIndex HuggingFace Embedding 适配器。

根据 DEV_SPEC 3.3.2：
- 支持本地模型（Sentence-Transformers, BGE）自由切换

本实现使用 LlamaIndex 的 HuggingFaceEmbedding，支持：
- Sentence-Transformers 模型
- BGE 模型
- 其他 HuggingFace 上的 Embedding 模型
"""

from typing import TYPE_CHECKING, List, Optional

from libs.embedding.base_embedding import BaseEmbedding, LLAMA_INDEX_AVAILABLE

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding as LlamaBaseEmbedding


class LocalEmbedding(BaseEmbedding):
    """本地 Embedding 实现，基于 LlamaIndex HuggingFace Embedding 适配器。

    默认使用 BAAI/bge-small-zh-v1.5（中文场景）或
    sentence-transformers/all-MiniLM-L6-v2（英文场景）。
    """

    provider_name: str = "local"
    default_model: str = "BAAI/bge-small-zh-v1.5"

    def get_llama_embedding(self) -> "LlamaBaseEmbedding":
        """获取 LlamaIndex HuggingFace Embedding 实例。"""
        if self._llama_embedding is not None:
            return self._llama_embedding

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-huggingface 未安装，"
                "请运行: pip install llama-index-embeddings-huggingface"
            )

        model_name = self.config.model or self.default_model

        kwargs = {
            "model_name": model_name,
        }

        # 可选：设置设备（cpu/cuda）
        # 默认会自动检测
        # kwargs["device"] = "cpu"

        self._llama_embedding = HuggingFaceEmbedding(**kwargs)
        return self._llama_embedding


class FakeEmbedding(BaseEmbedding):
    """测试用 Fake Embedding，返回固定维度零向量。"""

    provider_name: str = "fake"
    default_dimensions: int = 384

    def get_llama_embedding(self) -> "LlamaBaseEmbedding":
        """Fake Embedding 不使用 LlamaIndex，直接返回。"""
        raise NotImplementedError("FakeEmbedding 不支持 get_llama_embedding()")

    def embed(
        self,
        texts: List[str],
        trace=None,
    ) -> List[List[float]]:
        """返回固定维度零向量（仅用于测试）。"""
        if texts is None:
            raise ValueError(f"{self.provider_name} texts 不能为空")
        if not isinstance(texts, list):
            raise ValueError(f"{self.provider_name} texts 格式错误")
        for item in texts:
            if not isinstance(item, str):
                raise ValueError(f"{self.provider_name} texts 格式错误")

        if not texts:
            return []

        dim = self.config.dimensions or self.default_dimensions
        if dim <= 0:
            raise ValueError(f"{self.provider_name} dimensions 必须大于 0")
        return [[0.0 for _ in range(dim)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """返回固定维度零向量（仅用于测试）。"""
        dim = self.config.dimensions or self.default_dimensions
        return [0.0 for _ in range(dim)]

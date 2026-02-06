"""Qwen Embedding 实现，基于 LlamaIndex DashScope Embedding 适配器。

Qwen (DashScope) 提供原生 Embedding API，LlamaIndex 有专门的适配器支持。
支持的模型：text-embedding-v1, text-embedding-v2, text-embedding-v3 等。
"""

from typing import TYPE_CHECKING

from libs.embedding.base_embedding import BaseEmbedding, LLAMA_INDEX_AVAILABLE

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding as LlamaBaseEmbedding


class QwenEmbedding(BaseEmbedding):
    """Qwen Embedding 实现，基于 LlamaIndex DashScope 适配器。"""

    provider_name: str = "qwen"

    def get_llama_embedding(self) -> "LlamaBaseEmbedding":
        """获取 LlamaIndex DashScope Embedding 实例。"""
        if self._llama_embedding is not None:
            return self._llama_embedding

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.embeddings.dashscope import (
                DashScopeEmbedding,
                DashScopeTextEmbeddingModels,
                DashScopeTextEmbeddingType,
            )
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-dashscope 未安装，请运行: pip install llama-index-embeddings-dashscope"
            )

        # 构建参数
        kwargs = {
            "model_name": self.config.model,
            "api_key": self.config.api_key,
        }

        # DashScope 支持 text_type 参数（document 或 query）
        # 默认使用 document 类型
        if hasattr(DashScopeTextEmbeddingType, "TEXT_TYPE_DOCUMENT"):
            kwargs["text_type"] = DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT

        self._llama_embedding = DashScopeEmbedding(**kwargs)
        return self._llama_embedding

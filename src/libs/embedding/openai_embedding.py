"""OpenAI Embedding 实现，基于 LlamaIndex OpenAI Embedding 适配器。

根据 DEV_SPEC 3.3.3：
- Dense Embeddings（语义向量）：调用 Embedding 模型（如 OpenAI text-embedding-3）
  生成高维浮点向量，捕捉文本的深层语义关联
"""

from typing import TYPE_CHECKING, Optional

from libs.embedding.base_embedding import BaseEmbedding, LLAMA_INDEX_AVAILABLE

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding as LlamaBaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding 实现，基于 LlamaIndex OpenAI Embedding 适配器。"""

    provider_name: str = "openai"

    def get_llama_embedding(self) -> "LlamaBaseEmbedding":
        """获取 LlamaIndex OpenAI Embedding 实例。"""
        if self._llama_embedding is not None:
            return self._llama_embedding

        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core 未安装")

        try:
            from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-openai 未安装，请运行: pip install llama-index-embeddings-openai"
            )

        kwargs = {
            "model": self.config.model,
            "api_key": self.config.api_key,
        }

        # 可选参数
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions

        self._llama_embedding = LlamaOpenAIEmbedding(**kwargs)
        return self._llama_embedding

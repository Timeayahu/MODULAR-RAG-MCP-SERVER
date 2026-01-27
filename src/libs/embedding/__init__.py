"""
Embedding 抽象

向量化模型抽象层，支持 OpenAI Embedding、本地模型等多种向量化方案。
"""

from .base_embedding import BaseEmbedding
from .embedding_factory import EmbeddingFactory
from .local_embedding import LocalEmbedding

__all__ = ["BaseEmbedding", "EmbeddingFactory", "LocalEmbedding"]
"""Local Embedding 实现（占位）。"""

from typing import List, Optional

from libs.embedding.base_embedding import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    """本地 Embedding 实现（占位，返回固定维度向量）。"""

    provider_name: str = "local"
    default_dimensions: int = 384

    def embed(self, texts: List[str], trace=None) -> List[List[float]]:  # type: ignore[override]
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

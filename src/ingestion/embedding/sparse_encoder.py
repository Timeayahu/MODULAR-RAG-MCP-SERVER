"""SparseEncoder 实现 - 稀疏向量编码器（BM25）。

根据 DEV_SPEC 3.1.1 Embedding 阶段：
- Sparse Embeddings（稀疏向量）：利用 BM25 编码器生成稀疏向量（Keyword Weights），
  捕捉精确的关键词匹配信息，解决专有名词查找问题。
- 批处理优化：所有计算均采用 batch_size 驱动的批处理模式。

BM25 原理：
- TF (Term Frequency): 词频统计
- IDF (Inverse Document Frequency): 逆文档频率
- 输出：{term: weight} 映射，表示每个词的重要性权重
"""

import logging
import re
from collections import Counter, defaultdict
from math import log
from typing import TYPE_CHECKING, Dict, List, Optional

from ingestion.models import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# 默认停用词（简化版，实际应用中可扩展）
DEFAULT_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "can", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once",
}


class SparseEncoder:
    """稀疏向量编码器（基于 BM25）。

    特性：
    - 计算 BM25 term weights
    - 输出结构可用于 bm25_indexer
    - 支持停用词过滤
    - 处理空文本和特殊字符
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stop_words: Optional[set] = None,
    ) -> None:
        """初始化 SparseEncoder。

        Args:
            k1: BM25 k1 参数（控制词频饱和度）。
            b: BM25 b 参数（控制文档长度归一化）。
            stop_words: 停用词集合，如果为 None 则使用默认停用词。
        """
        self.k1 = k1
        self.b = b
        self.stop_words = stop_words if stop_words is not None else DEFAULT_STOP_WORDS

        # 文档统计信息（在 fit 时计算）
        self._doc_count = 0
        self._avg_doc_length = 0.0
        self._doc_freq: Dict[str, int] = defaultdict(int)  # 每个词出现在多少文档中
        self._fitted = False

    def fit(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> "SparseEncoder":
        """在 chunks 上拟合 IDF 统计。

        Args:
            chunks: Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            self（支持链式调用）。
        """
        if not chunks:
            logger.warning("fit 接收到空 chunks 列表")
            self._fitted = True
            return self

        # 收集文档长度和词频
        doc_lengths = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            doc_lengths.append(len(tokens))

            # 统计每个词出现在多少文档中（用于 IDF）
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freq[token] += 1

        self._doc_count = len(chunks)
        self._avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        self._fitted = True

        logger.debug(
            f"SparseEncoder fitted: {self._doc_count} docs, "
            f"avg_length={self._avg_doc_length:.2f}, "
            f"vocab_size={len(self._doc_freq)}"
        )

        return self

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional["TraceContext"] = None,
    ) -> List[Dict[str, float]]:
        """将 Chunks 编码为稀疏向量（term weights）。

        Args:
            chunks: Chunk 列表。
            trace: 可选追踪上下文。

        Returns:
            稀疏向量列表，每个元素是 {term: weight} 字典。

        Raises:
            ValueError: 如果 chunks 为空或未 fit。
        """
        if not chunks:
            raise ValueError("chunks 不能为空")

        if not self._fitted:
            logger.warning("SparseEncoder 未 fit，将先进行 fit")
            self.fit(chunks, trace=trace)

        sparse_vectors = []
        for chunk in chunks:
            term_weights = self._compute_bm25_weights(chunk.text)
            sparse_vectors.append(term_weights)

        return sparse_vectors

    def encode_single(
        self,
        chunk: Chunk,
        trace: Optional["TraceContext"] = None,
    ) -> Dict[str, float]:
        """编码单个 Chunk。

        Args:
            chunk: 单个 Chunk。
            trace: 可选追踪上下文。

        Returns:
            稀疏向量 {term: weight}。
        """
        vectors = self.encode([chunk], trace=trace)
        return vectors[0]

    def _tokenize(self, text: str) -> List[str]:
        """分词并清理文本。

        Args:
            text: 输入文本。

        Returns:
            Token 列表。
        """
        if not text or not text.strip():
            return []

        # 转小写
        text = text.lower()

        # 替换连字符和下划线为空格
        text = re.sub(r'[-_]', ' ', text)

        # 简单分词（按空格和标点）
        tokens = re.findall(r'\b\w+\b', text)

        # 过滤停用词和短词
        tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

        return tokens

    def _compute_bm25_weights(self, text: str) -> Dict[str, float]:
        """计算 BM25 term weights。

        Args:
            text: 输入文本。

        Returns:
            {term: weight} 字典。
        """
        tokens = self._tokenize(text)

        if not tokens:
            return {}

        # 计算词频
        term_freq = Counter(tokens)
        doc_length = len(tokens)

        # 计算 BM25 权重
        weights = {}
        for term, tf in term_freq.items():
            # IDF 计算
            df = self._doc_freq.get(term, 0)
            if df == 0:
                # 未见过的词，给一个小的 IDF
                idf = log((self._doc_count + 1) / 1)
            else:
                idf = log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 公式
            norm_factor = 1 - self.b + self.b * (doc_length / self._avg_doc_length) if self._avg_doc_length > 0 else 1
            bm25_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)

            weights[term] = bm25_score

        return weights

    @property
    def is_fitted(self) -> bool:
        """返回是否已拟合。"""
        return self._fitted

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小。"""
        return len(self._doc_freq)

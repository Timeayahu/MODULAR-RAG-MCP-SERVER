"""Query Engine 核心数据模型。

定义查询处理和检索过程中使用的数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProcessedQuery:
    """处理后的查询对象。
    
    Attributes:
        original_query: 原始查询字符串。
        keywords: 提取的关键词列表（用于 BM25 等稀疏检索）。
        filters: 元数据过滤条件字典（用于向量库过滤）。
        metadata: 额外的处理元数据（例如停用词数量、处理方法等）。
    """
    
    original_query: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict。
        
        Returns:
            字典表示。
        """
        return {
            "original_query": self.original_query,
            "keywords": self.keywords,
            "filters": self.filters,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalCandidate:
    """检索候选结果。
    
    Attributes:
        chunk_id: Chunk 唯一标识。
        text: Chunk 文本内容。
        score: 相关性分数。
        metadata: 元数据字典。
        source: 来源标识（dense/sparse/fusion/rerank）。
    """
    
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    
    #手写todict更灵活，可以选择性序列化部分字段，还能添加额外的逻辑。asdict不行 
    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict。
        
        Returns:
            字典表示。
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }

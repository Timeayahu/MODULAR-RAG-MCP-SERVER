"""Query Processor - 查询预处理模块。

负责对原始查询进行预处理，提取关键词并解析过滤条件。
根据 DEV_SPEC Task D1：
- 关键词提取（先规则/分词方式）
- 支持通用 filters 结构（dict 格式）
"""

import logging
import re
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Any

import jieba

from core.query_engine.models import ProcessedQuery

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class QueryProcessor:
    """查询预处理器。
    
    负责将原始查询字符串转换为结构化的 ProcessedQuery 对象。
    
    设计原则：
    - 单一职责：只做查询预处理，不涉及检索逻辑
    - 配置驱动：停用词、分词策略可配置
    - 可观测：支持 TraceContext（预留）
    
    Attributes:
        stopwords: 停用词集合。
        min_keyword_length: 最小关键词长度。
    """
    
    # 默认中文停用词（常见虚词、助词、连词）
    DEFAULT_CHINESE_STOPWORDS = {
        "的", "了", "和", "是", "在", "有", "我", "他", "她", "它", "们",
        "这", "那", "都", "也", "就", "要", "会", "能", "可以", "吗", "呢",
        "啊", "吧", "什么", "怎么", "为什么", "哪里", "哪个", "多少",
        "一个", "一些", "一样", "不是", "没有", "没", "不", "非常", "很",
    }
    
    # 默认英文停用词（常见介词、冠词、连词）
    DEFAULT_ENGLISH_STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "about", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "what", "which", "who", "whom", "this", "that", "these", "those",
    }
    
    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        min_keyword_length: int = 1,
    ):
        """初始化 QueryProcessor。
        
        Args:
            stopwords: 自定义停用词集合，None 则使用默认停用词。
            min_keyword_length: 最小关键词长度（字符数），默认 1（支持中文单字）。
        """
        if stopwords is None:
            # 合并中英文默认停用词
            self.stopwords = self.DEFAULT_CHINESE_STOPWORDS | self.DEFAULT_ENGLISH_STOPWORDS
        else:
            self.stopwords = stopwords
        
        self.min_keyword_length = min_keyword_length
        
        logger.debug(
            f"QueryProcessor initialized with {len(self.stopwords)} stopwords, "
            f"min_keyword_length={min_keyword_length}"
        )
    
    def process(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional["TraceContext"] = None,
    ) -> ProcessedQuery:
        """处理原始查询。
        
        Args:
            query: 原始查询字符串。
            filters: 可选的元数据过滤条件。
            trace: 可选的追踪上下文（预留）。
        
        Returns:
            ProcessedQuery 对象。
        
        Raises:
            ValueError: 如果查询为空或无效。
        """
        # 输入校验
        if not query or not query.strip():
            raise ValueError("查询字符串不能为空")
        
        query = query.strip()
        
        # 提取关键词
        keywords = self._extract_keywords(query)
        
        # 解析过滤条件
        processed_filters = filters if filters is not None else {}
        
        # 构建处理元数据
        metadata = {
            "method": "rule_based",
            "stopwords_removed": len(self._tokenize(query)) - len(keywords),
            "original_length": len(query),
        }
        
        # 可观测：记录处理阶段（预留）
        if trace:
            trace.record_stage(
                stage="query_processing",
                details={
                    "original_query": query,
                    "keywords_count": len(keywords),
                    "filters": processed_filters,
                },
            )
        
        logger.info(
            f"Query processed: '{query[:50]}...' -> {len(keywords)} keywords, "
            f"{len(processed_filters)} filters"
        )
        
        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=processed_filters,
            metadata=metadata,
        )
    
    def _extract_keywords(self, query: str) -> List[str]:#方法名前加 _ 表示该方法是约定俗成的"private"，如果一个方法只是为了支撑其他主要功能的实现，而不应该作为独立功能暴露给用户，就加下划线。
        """从查询中提取关键词。
        
        采用 jieba 分词方法：
        1. 分词（使用 jieba 进行中文分词，保持完整词汇）
        2. 去除停用词
        3. 过滤短词
        4. 去重并保持顺序
        
        Args:
            query: 查询字符串。
        
        Returns:
            关键词列表（去重、保序）。
        """
        tokens = self._tokenize(query)
        
        # 过滤：去停用词 + 长度检查
        keywords = []
        seen = set()
        
        for token in tokens:
            token_lower = token.lower()
            
            # 跳过停用词
            if token_lower in self.stopwords:
                continue
            
            # 跳过过短词
            if len(token) < self.min_keyword_length:
                continue
            
            # 去重（保持首次出现顺序）
            if token_lower not in seen:
                keywords.append(token)
                seen.add(token_lower)
        
        return keywords
    
    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 进行中文分词。
        
        规则：
        - 中文：使用 jieba 分词，保持完整词汇
        - 英文：按空格和标点切分
        - 数字：保留
        
        Args:
            text: 输入文本。
        
        Returns:
            Token 列表。
        """
        # 使用 jieba 进行分词，这会正确处理中文词汇
        # jieba.cut() 返回生成器，转换为列表
        tokens = list(jieba.cut(text))
        
        # 过滤空白字符和纯标点符号
        result = []
        for token in tokens:
            token = token.strip()
            if token and not re.match(r'^[^\w\u4e00-\u9fff]+$', token):
                result.append(token)
        
        return result
    
    def add_stopwords(self, words: Set[str]) -> None:
        """添加自定义停用词。
        
        Args:
            words: 停用词集合。
        """
        self.stopwords.update(words)
        logger.debug(f"Added {len(words)} custom stopwords")
    
    def remove_stopwords(self, words: Set[str]) -> None:
        """移除停用词。
        
        Args:
            words: 要移除的停用词集合。
        """
        self.stopwords -= words
        logger.debug(f"Removed {len(words)} stopwords")

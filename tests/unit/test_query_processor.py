"""QueryProcessor 单元测试。

测试查询预处理功能，包括关键词提取和过滤条件解析。

pytest -p no:logfire tests\unit\test_query_processor.py
"""

import sys
from pathlib import Path

# 添加 src 到路径
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import pytest
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.models import ProcessedQuery


class TestQueryProcessor:
    """QueryProcessor 测试套件。"""
    
    def test_init_default_stopwords(self):
        """测试：使用默认停用词初始化。"""
        processor = QueryProcessor()
        
        assert len(processor.stopwords) > 0
        assert "的" in processor.stopwords  # 中文停用词
        assert "the" in processor.stopwords  # 英文停用词
    
    def test_init_custom_stopwords(self):
        """测试：使用自定义停用词初始化。"""
        custom_stopwords = {"测试", "test"}
        processor = QueryProcessor(stopwords=custom_stopwords)
        
        assert processor.stopwords == custom_stopwords
        assert "测试" in processor.stopwords
    
    def test_process_simple_chinese_query(self):
        """测试：处理简单中文查询。"""
        processor = QueryProcessor()
        result = processor.process("什么是人工智能")
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == "什么是人工智能"
        assert len(result.keywords) > 0
        
        # "什么" 和 "是" 是停用词，应该被过滤
        assert "什么" not in result.keywords
        assert "是" not in result.keywords
        
        # "机器学习"作为一个完整的keyword被jieba识别
        assert "人工智能" in result.keywords
    
    def test_process_simple_english_query(self):
        """测试：处理简单英文查询。"""
        processor = QueryProcessor()
        result = processor.process("What is machine learning")
        
        assert result.original_query == "What is machine learning"
        assert len(result.keywords) > 0
        
        # 停用词应该被过滤
        assert "what" not in [k.lower() for k in result.keywords]
        assert "is" not in [k.lower() for k in result.keywords]
        
        # 关键词应该被保留
        assert "machine" in [k.lower() for k in result.keywords]
        assert "learning" in [k.lower() for k in result.keywords]
    
    def test_process_mixed_language_query(self):
        """测试：处理中英文混合查询。"""
        processor = QueryProcessor()
        result = processor.process("RAG技术在NLP中的应用")
        
        assert len(result.keywords) > 0
        
        # 英文词应该被保留
        assert "RAG" in result.keywords
        assert "NLP" in result.keywords
        
        # 中文字符应该被提取
        assert "技术" in result.keywords
    
    def test_process_with_numbers(self):
        """测试：处理包含数字的查询。"""
        processor = QueryProcessor()
        result = processor.process("2024年人工智能发展趋势")
        
        # 数字应该被保留
        assert "2024" in result.keywords
    
    def test_process_with_filters(self):
        """测试：处理带过滤条件的查询。"""
        processor = QueryProcessor()
        filters = {
            "collection": "papers",
            "year": 2024,
        }
        
        result = processor.process("机器学习", filters=filters)
        
        assert result.filters == filters
        assert result.filters["collection"] == "papers"
        assert result.filters["year"] == 2024
    
    def test_process_empty_query_raises_error(self):
        """测试：空查询应该抛出错误。"""
        processor = QueryProcessor()
        
        with pytest.raises(ValueError, match="查询字符串不能为空"):
            processor.process("")
        
        with pytest.raises(ValueError, match="查询字符串不能为空"):
            processor.process("   ")
    
    def test_keywords_deduplication(self):
        """测试：关键词去重（保持顺序）。"""
        processor = QueryProcessor()
        result = processor.process("machine learning and machine translation")
        
        # "machine" 只应该出现一次（首次出现）
        machine_count = sum(1 for k in result.keywords if k.lower() == "machine")
        assert machine_count == 1
    
    def test_min_keyword_length(self):
        """测试：最小关键词长度过滤。"""
        processor = QueryProcessor(min_keyword_length=3)
        result = processor.process("AI is ML")
        
        # "AI" 和 "ML" 长度 < 3，应该被过滤
        # "is" 是停用词，也会被过滤
        # 结果应该为空或只包含长度 >= 3 的词
        for keyword in result.keywords:
            assert len(keyword) >= 3
    
    def test_add_stopwords(self):
        """测试：动态添加停用词。"""
        processor = QueryProcessor()
        
        # 初始状态："技术" 不是停用词
        result1 = processor.process("我特么的真的是一个大煞笔")
        initial_count = len(result1.keywords)
        
        # 添加 "技术" 为停用词
        processor.add_stopwords({"煞笔", "特么的"})
        
        result2 = processor.process("我特么的真的是一个大煞笔")
        
        # 关键词数量应该减少
        assert len(result2.keywords) < initial_count
    
    def test_remove_stopwords(self):
        """测试：动态移除停用词。"""
        processor = QueryProcessor()
        
        # 添加自定义停用词（单字）
        processor.add_stopwords({"烧杯"})
        
        result1 = processor.process("小烧杯一个，一个小烧杯啊")
        # "重" 和 "要" 都应该被过滤
        assert "烧杯" not in result1.keywords
      
        
        # 移除停用词
        processor.remove_stopwords({"烧杯"})
        
        result2 = processor.process("小烧杯一个，一个小烧杯啊")
        # 移除后应该能提取到
        assert "烧杯" in result2.keywords
        
    
    def test_metadata_populated(self):
        """测试：元数据字段被正确填充。"""
        processor = QueryProcessor()
        result = processor.process("什么是机器学习")
        
        assert "method" in result.metadata
        assert result.metadata["method"] == "rule_based"
        
        assert "stopwords_removed" in result.metadata
        assert isinstance(result.metadata["stopwords_removed"], int)
        
        assert "original_length" in result.metadata
        assert result.metadata["original_length"] == len("什么是机器学习")
    
    def test_to_dict(self):
        """测试：ProcessedQuery 序列化。"""
        processor = QueryProcessor()
        result = processor.process("测试查询")
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "original_query" in result_dict
        assert "keywords" in result_dict
        assert "filters" in result_dict
        assert "metadata" in result_dict


class TestTokenization:
    """分词功能测试。"""
    
    def test_tokenize_chinese(self):
        """测试：中文分词。"""
        processor = QueryProcessor()
        tokens = processor._tokenize("我已经成功修改了代码")
        
        # 中文应该按字拆分
        assert "成功" in tokens
        assert "修改" in tokens
        assert "代码" in tokens

    
    def test_tokenize_english(self):
        """测试：英文分词。"""
        processor = QueryProcessor()
        tokens = processor._tokenize("machine learning")
        
        # 英文按词保留
        assert "machine" in tokens
        assert "learning" in tokens
    
    def test_tokenize_mixed(self):
        """测试：中英文混合分词。"""
        processor = QueryProcessor()
        tokens = processor._tokenize("使用Python进行数据分析")
        
        # 中文字符
        assert "使用" in tokens
        assert "数据分析" in tokens
        
        # 英文词
        assert "Python" in tokens
    
    def test_tokenize_with_punctuation(self):
        """测试：包含标点符号的分词。"""
        processor = QueryProcessor()
        tokens = processor._tokenize("Hello, world! 你好，世界！")
        
        # 标点应该被忽略
        assert "," not in tokens
        assert "!" not in tokens
        assert "，" not in tokens
        assert "！" not in tokens
        
        # 词和字应该被保留
        assert "Hello" in tokens
        assert "world" in tokens
        assert "你好" in tokens


class TestEdgeCases:
    """边界情况测试。"""
    
    def test_query_with_only_stopwords(self):
        """测试：只包含停用词的查询。"""
        processor = QueryProcessor()
        result = processor.process("的 是 在 有")
        
        # 所有词都是停用词，关键词列表应该为空
        assert len(result.keywords) == 0
    
    def test_query_with_special_characters(self):
        """测试：包含特殊字符的查询。"""
        processor = QueryProcessor()
        result = processor.process("@#$%^&*()")
        
        # 特殊字符应该被过滤
        assert len(result.keywords) == 0
    
    def test_very_long_query(self):
        """测试：非常长的查询。"""
        processor = QueryProcessor()
        long_query = "机器学习 " * 100
        
        result = processor.process(long_query)
        
        # 应该能正常处理
        assert isinstance(result, ProcessedQuery)
        assert len(result.keywords) > 0
    
    def test_query_with_whitespace(self):
        """测试：包含多余空格的查询。"""
        processor = QueryProcessor()
        result = processor.process("  机器学习   ")
        
        # 空格应该被正确处理
        assert result.original_query == "机器学习"
        assert len(result.keywords) > 0

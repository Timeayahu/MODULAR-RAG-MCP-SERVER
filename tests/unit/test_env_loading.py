"""
测试环境变量加载功能
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 添加 src 到 Python 路径
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_env_file_loading():
    """测试 .env 文件是否被正确加载"""
    # 创建临时 .env 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("TEST_API_KEY=test_value_12345\n")
        f.write("TEST_BASE_URL=https://test.example.com\n")
        temp_env_path = f.name
    
    try:
        # 加载 .env 文件
        from dotenv import load_dotenv
        load_dotenv(temp_env_path)
        
        # 验证环境变量是否被加载
        assert os.getenv("TEST_API_KEY") == "test_value_12345"
        assert os.getenv("TEST_BASE_URL") == "https://test.example.com"
    finally:
        # 清理
        os.unlink(temp_env_path)
        if "TEST_API_KEY" in os.environ:
            del os.environ["TEST_API_KEY"]
        if "TEST_BASE_URL" in os.environ:
            del os.environ["TEST_BASE_URL"]


def test_env_var_expansion_in_settings():
    """测试 settings.py 中的环境变量替换功能"""
    from core.settings import _expand_env_vars
    
    # 设置测试环境变量
    os.environ["TEST_VAR_1"] = "value1"
    os.environ["TEST_VAR_2"] = "value2"
    
    try:
        # 测试字符串替换
        assert _expand_env_vars("${TEST_VAR_1}") == "value1"
        assert _expand_env_vars("${TEST_VAR_2}") == "value2"
        
        # 测试嵌套在字典中的替换
        data = {
            "api_key": "${TEST_VAR_1}",
            "base_url": "https://example.com",
            "nested": {
                "secret": "${TEST_VAR_2}"
            }
        }
        result = _expand_env_vars(data)
        assert result["api_key"] == "value1"
        assert result["base_url"] == "https://example.com"
        assert result["nested"]["secret"] == "value2"
        
        # 测试列表中的替换
        data_list = ["${TEST_VAR_1}", "normal_value", "${TEST_VAR_2}"]
        result_list = _expand_env_vars(data_list)
        assert result_list == ["value1", "normal_value", "value2"]
        
    finally:
        # 清理
        if "TEST_VAR_1" in os.environ:
            del os.environ["TEST_VAR_1"]
        if "TEST_VAR_2" in os.environ:
            del os.environ["TEST_VAR_2"]


def test_env_var_not_set():
    """测试未设置的环境变量返回 None"""
    from core.settings import _expand_env_vars
    
    # 确保环境变量不存在
    if "NONEXISTENT_VAR" in os.environ:
        del os.environ["NONEXISTENT_VAR"]
    
    result = _expand_env_vars("${NONEXISTENT_VAR}")
    assert result is None

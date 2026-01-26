"""
结构化日志模块

提供统一的日志接口，支持 JSON 格式输出。
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    获取配置好的日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 创建控制台处理器，输出到 stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logger.level)
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    # 防止日志传播到根日志器
    logger.propagate = False
    
    return logger
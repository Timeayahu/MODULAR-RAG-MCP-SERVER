#!/usr/bin/env python3
"""
Modular RAG MCP Server 主入口

这是 MCP Server 的启动入口，负责：
1. 加载配置文件
2. 初始化日志系统
3. 启动 MCP Server (Stdio Transport)
"""

import sys
from pathlib import Path

# 添加 src 到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.settings import load_settings
from observability.logger import get_logger


def main():
    """主入口函数"""
    try:
        # 加载配置
        settings = load_settings("config/settings.yaml")
        
        # 初始化日志
        logger = get_logger(__name__)
        logger.info("Modular RAG MCP Server 启动中...")
        
        # TODO: 在后续阶段启动 MCP Server
        logger.info("配置加载成功，等待 MCP Server 实现...")
        
    except Exception as e:
        print(f"启动失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Ingestion Pipeline

离线数据摄取管道，负责文档加载、处理、向量化和存储。
"""

from .models import Chunk, Document

__all__ = ["Document", "Chunk"]
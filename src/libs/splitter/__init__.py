"""
Splitter 抽象

文本切分器抽象层，支持递归切分、语义切分、定长切分等多种策略。
"""

from .base_splitter import BaseSplitter
from .recursive_splitter import RecursiveSplitter
from .splitter_factory import SplitterFactory

__all__ = ["BaseSplitter", "RecursiveSplitter", "SplitterFactory"]
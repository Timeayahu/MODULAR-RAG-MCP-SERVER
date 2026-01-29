#!/usr/bin/env python3
"""
数据摄取脚本 (Data Ingestion Script)

用法:
    python scripts/ingest.py --path <file_or_dir> [--collection <name>] [--force]

示例:
    # 摄取单个文件
    python scripts/ingest.py --path data/documents/sample.pdf --collection papers

    # 摄取整个目录
    python scripts/ingest.py --path data/documents/papers/ --collection papers

    # 强制重新处理（即使文件未变更）
    python scripts/ingest.py --path data/documents/sample.pdf --force
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],  # 日志输出到 stderr
)
logger = logging.getLogger(__name__)


def collect_files(path: Path) -> List[Path]:
    """收集待处理的文件。

    Args:
        path: 文件或目录路径。

    Returns:
        文件路径列表。

    Raises:
        FileNotFoundError: 如果路径不存在。
        ValueError: 如果路径既不是文件也不是目录。
    """
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if path.is_file():
        return [path]
    elif path.is_dir():
        # 收集目录下所有 PDF 文件（递归）
        files = list(path.rglob("*.pdf"))
        if not files:
            logger.warning(f"目录 {path} 中未找到 PDF 文件。")
        return files
    else:
        raise ValueError(f"路径既不是文件也不是目录: {path}")


def main():
    """脚本主入口。"""
    parser = argparse.ArgumentParser(
        description="数据摄取脚本 - 将文档导入知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --path data/documents/sample.pdf --collection papers
  %(prog)s --path data/documents/papers/ --collection papers --force
        """,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="待处理的文件或目录路径",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="集合名称（默认: default）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理，即使文件未变更",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径（默认: config/settings.yaml）",
    )

    args = parser.parse_args()

    # 加载配置
    try:
        settings = load_settings(args.config)
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)

    # 收集待处理文件
    try:
        path = Path(args.path)
        files = collect_files(path)
        logger.info(f"找到 {len(files)} 个文件待处理。")
    except Exception as e:
        logger.error(f"收集文件失败: {e}")
        sys.exit(1)

    if not files:
        logger.warning("没有文件需要处理。")
        sys.exit(0)

    # 创建 Ingestion Pipeline（按集合名称初始化）
    try:
        pipeline = IngestionPipeline(settings, collection=args.collection)
    except Exception as e:
        logger.error(f"创建 Pipeline 失败: {e}")
        sys.exit(1)

    # 处理每个文件
    total_files = len(files)
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, file_path in enumerate(files, 1):
        logger.info(f"[{i}/{total_files}] 处理文件: {file_path}")
        try:
            # IngestionPipeline.run 只接受 path 和 force 参数
            result = pipeline.run(str(file_path), force=args.force)

            if result.get("skipped"):
                logger.info(f"文件 {file_path.name} 未变更，已跳过。")
                skip_count += 1
            else:
                num_chunks = result.get("num_chunks", 0)
                logger.info(f"文件 {file_path.name} 处理成功，生成 {num_chunks} 个 Chunk。")
                success_count += 1

        except Exception as e:
            logger.error(f"文件 {file_path.name} 处理失败: {e}")
            error_count += 1

    # 输出摘要
    logger.info("=" * 60)
    logger.info("摄取完成！")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功处理: {success_count}")
    logger.info(f"跳过: {skip_count}")
    logger.info(f"失败: {error_count}")
    logger.info("=" * 60)

    # 根据结果设置退出码
    if error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

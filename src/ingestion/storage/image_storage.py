"""ImageStorage 实现 - 图片文件存储器。

根据 DEV_SPEC 3.5.3 Storage 阶段：
- 原始图片存储：将提取的图片保存至本地文件系统的约定目录
  （如 data/images/{collection}/{image_id}.png）
- 索引表：记录每张图片的 image_id、file_path、source_doc、page 等信息
- 检索命中后，根据 Chunk 的 image_refs 查询索引表，获取图片文件路径用于返回
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ImageStorage:
    """图片文件存储器。

    特性：
    - 保存图片到文件系统
    - 维护 image_id -> file_path 映射
    - 支持多个 collection
    - JSON 格式索引（可扩展为 SQLite）
    """

    def __init__(
        self,
        storage_dir: str = "data/images",
        index_file: str = "data/images/index.json",
    ) -> None:
        """初始化 ImageStorage。

        Args:
            storage_dir: 图片存储根目录。
            index_file: 索引文件路径。
        """
        self.storage_dir = Path(storage_dir)
        self.index_file = Path(index_file)

        # 确保目录存在
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载索引
        self.index: Dict[str, Dict] = self._load_index()

    def save_image(
        self,
        image_id: str,
        image_data: bytes,
        collection: str = "default",
        metadata: Optional[Dict] = None,
        extension: str = ".png",
    ) -> str:
        """保存图片到文件系统。

        Args:
            image_id: 图片唯一标识。
            image_data: 图片二进制数据。
            collection: 集合名称。
            metadata: 可选的图片元数据（source_doc, page 等）。
            extension: 文件扩展名。

        Returns:
            保存的文件绝对路径。

        Raises:
            ValueError: 如果 image_data 为空。
        """
        if not image_data:
            raise ValueError("image_data 不能为空")

        # 创建 collection 目录
        collection_dir = self.storage_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        filename = f"{image_id}{extension}"
        file_path = collection_dir / filename

        # 保存图片
        try:
            with open(file_path, "wb") as f:
                f.write(image_data)
            logger.debug(f"图片已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            raise

        # 更新索引（使用绝对路径）
        absolute_path = str(file_path.absolute())
        self.index[image_id] = {
            "file_path": absolute_path,
            "collection": collection,
            "metadata": metadata or {},
        }

        # 持久化索引
        self._save_index()

        return absolute_path

    def get_image_path(self, image_id: str) -> Optional[str]:
        """获取图片文件路径。

        Args:
            image_id: 图片 ID。

        Returns:
            图片文件路径，如果不存在则返回 None。
        """
        if image_id not in self.index:
            return None

        return self.index[image_id]["file_path"]

    def get_image_info(self, image_id: str) -> Optional[Dict]:
        """获取图片完整信息。

        Args:
            image_id: 图片 ID。

        Returns:
            图片信息字典，如果不存在则返回 None。
        """
        return self.index.get(image_id)

    def image_exists(self, image_id: str) -> bool:
        """检查图片是否存在。

        Args:
            image_id: 图片 ID。

        Returns:
            如果图片存在返回 True，否则返回 False。
        """
        if image_id not in self.index:
            return False

        # 检查文件是否真实存在
        file_path = Path(self.index[image_id]["file_path"])
        return file_path.exists()

    def delete_image(self, image_id: str) -> bool:
        """删除图片。

        Args:
            image_id: 图片 ID。

        Returns:
            如果删除成功返回 True，如果图片不存在返回 False。
        """
        if image_id not in self.index:
            return False

        # 删除文件
        file_path = Path(self.index[image_id]["file_path"])
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"图片已删除: {file_path}")
        except Exception as e:
            logger.error(f"删除图片失败: {e}")
            raise

        # 从索引中移除
        del self.index[image_id]
        self._save_index()

        return True

    def list_images(self, collection: Optional[str] = None) -> List[str]:
        """列出图片 ID。

        Args:
            collection: 可选的集合名称，如果指定则只返回该集合的图片。

        Returns:
            图片 ID 列表。
        """
        if collection is None:
            return list(self.index.keys())

        return [
            image_id
            for image_id, info in self.index.items()
            if info["collection"] == collection
        ]

    def clear_collection(self, collection: str) -> int:
        """清空指定集合的所有图片。

        Args:
            collection: 集合名称。

        Returns:
            删除的图片数量。
        """
        image_ids = self.list_images(collection)
        count = 0

        for image_id in image_ids:
            if self.delete_image(image_id):
                count += 1

        # 删除 collection 目录
        collection_dir = self.storage_dir / collection
        if collection_dir.exists() and not any(collection_dir.iterdir()):
            collection_dir.rmdir()
            logger.debug(f"集合目录已删除: {collection_dir}")

        return count

    def _load_index(self) -> Dict[str, Dict]:
        """加载索引文件。

        Returns:
            索引字典。
        """
        if not self.index_file.exists():
            return {}

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载索引文件失败: {e}")
            return {}

    def _save_index(self) -> None:
        """保存索引文件。"""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}")
            raise

    @property
    def num_images(self) -> int:
        """返回索引中的图片总数。"""
        return len(self.index)

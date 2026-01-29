"""ImageStorage 测试。

根据 DEV_SPEC C13 验收标准：
- 保存后文件存在
- 查找 image_id 返回正确路径
"""

import shutil
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def temp_storage_dir(tmp_path):
    """创建临时存储目录。"""
    storage_dir = tmp_path / "images"
    index_file = tmp_path / "images" / "index.json"
    yield str(storage_dir), str(index_file)
    # 清理
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def sample_image_data():
    """创建示例图片数据。"""
    # 简单的二进制数据模拟图片
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00"


@pytest.mark.unit
def test_image_storage_save_image(temp_storage_dir, sample_image_data):
    """验收标准：保存后文件存在。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    image_id = "img_001"
    collection = "test_collection"

    # 保存图片
    file_path = storage.save_image(
        image_id,
        sample_image_data,
        collection=collection,
    )

    # 验证文件存在
    assert file_path is not None
    full_path = Path(file_path)
    assert full_path.exists()

    # 验证文件内容
    with open(full_path, "rb") as f:
        saved_data = f.read()
    assert saved_data == sample_image_data


@pytest.mark.unit
def test_image_storage_get_image_path(temp_storage_dir, sample_image_data):
    """验收标准：查找 image_id 返回正确路径。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    image_id = "img_002"
    saved_path = storage.save_image(image_id, sample_image_data)

    # 查找图片路径
    retrieved_path = storage.get_image_path(image_id)

    # 应返回相同的路径
    assert retrieved_path == saved_path


@pytest.mark.unit
def test_image_storage_nonexistent_image(temp_storage_dir):
    """查找不存在的图片应返回 None。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    # 查找不存在的图片
    path = storage.get_image_path("nonexistent")
    assert path is None


@pytest.mark.unit
def test_image_storage_image_exists(temp_storage_dir, sample_image_data):
    """测试 image_exists 方法。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    image_id = "img_003"

    # 保存前不存在
    assert not storage.image_exists(image_id)

    # 保存后存在
    storage.save_image(image_id, sample_image_data)
    assert storage.image_exists(image_id)


@pytest.mark.unit
def test_image_storage_with_metadata(temp_storage_dir, sample_image_data):
    """测试保存图片时附带元数据。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    image_id = "img_004"
    metadata = {
        "source_doc": "doc1.pdf",
        "page": 5,
        "caption": "System architecture diagram",
    }

    storage.save_image(
        image_id,
        sample_image_data,
        metadata=metadata,
    )

    # 获取图片信息
    info = storage.get_image_info(image_id)
    assert info is not None
    assert info["metadata"] == metadata


@pytest.mark.unit
def test_image_storage_delete_image(temp_storage_dir, sample_image_data):
    """测试删除图片。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    image_id = "img_005"
    file_path = storage.save_image(image_id, sample_image_data)

    # 验证文件存在
    assert Path(file_path).exists()
    assert storage.image_exists(image_id)

    # 删除图片
    result = storage.delete_image(image_id)
    assert result is True

    # 验证文件不存在
    assert not Path(file_path).exists()
    assert not storage.image_exists(image_id)


@pytest.mark.unit
def test_image_storage_delete_nonexistent(temp_storage_dir):
    """删除不存在的图片应返回 False。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    result = storage.delete_image("nonexistent")
    assert result is False


@pytest.mark.unit
def test_image_storage_list_images(temp_storage_dir, sample_image_data):
    """测试列出图片。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    # 保存多个图片到不同集合
    storage.save_image("img_1", sample_image_data, collection="col1")
    storage.save_image("img_2", sample_image_data, collection="col1")
    storage.save_image("img_3", sample_image_data, collection="col2")

    # 列出所有图片
    all_images = storage.list_images()
    assert len(all_images) == 3
    assert "img_1" in all_images
    assert "img_2" in all_images
    assert "img_3" in all_images

    # 列出特定集合的图片
    col1_images = storage.list_images(collection="col1")
    assert len(col1_images) == 2
    assert "img_1" in col1_images
    assert "img_2" in col1_images


@pytest.mark.unit
def test_image_storage_clear_collection(temp_storage_dir, sample_image_data):
    """测试清空集合。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    # 保存图片到两个集合
    storage.save_image("img_1", sample_image_data, collection="col1")
    storage.save_image("img_2", sample_image_data, collection="col1")
    storage.save_image("img_3", sample_image_data, collection="col2")

    # 清空 col1
    count = storage.clear_collection("col1")
    assert count == 2

    # 验证 col1 的图片被删除
    assert not storage.image_exists("img_1")
    assert not storage.image_exists("img_2")

    # 验证 col2 的图片仍存在
    assert storage.image_exists("img_3")


@pytest.mark.unit
def test_image_storage_custom_extension(temp_storage_dir, sample_image_data):
    """测试自定义文件扩展名。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    image_id = "img_006"
    file_path = storage.save_image(
        image_id,
        sample_image_data,
        extension=".jpg",
    )

    # 验证文件扩展名
    assert file_path.endswith(".jpg")


@pytest.mark.unit
def test_image_storage_empty_data_raises_error(temp_storage_dir):
    """空图片数据应抛出错误。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    with pytest.raises(ValueError, match="image_data 不能为空"):
        storage.save_image("img", b"")


@pytest.mark.unit
def test_image_storage_persistence(temp_storage_dir, sample_image_data):
    """测试索引持久化。"""
    storage_dir, index_file = temp_storage_dir

    # 创建第一个实例并保存图片
    storage1 = ImageStorage(storage_dir, index_file)
    storage1.save_image("img_007", sample_image_data, collection="col1")

    # 创建第二个实例（模拟重启）
    storage2 = ImageStorage(storage_dir, index_file)

    # 应该能找到之前保存的图片
    assert storage2.image_exists("img_007")
    path = storage2.get_image_path("img_007")
    assert path is not None


@pytest.mark.unit
def test_image_storage_num_images_property(temp_storage_dir, sample_image_data):
    """测试 num_images 属性。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    assert storage.num_images == 0

    storage.save_image("img_1", sample_image_data)
    assert storage.num_images == 1

    storage.save_image("img_2", sample_image_data)
    assert storage.num_images == 2

    storage.delete_image("img_1")
    assert storage.num_images == 1


@pytest.mark.unit
def test_image_storage_multiple_collections(temp_storage_dir, sample_image_data):
    """测试多个集合的图片存储。"""
    storage_dir, index_file = temp_storage_dir
    storage = ImageStorage(storage_dir, index_file)

    # 保存到不同集合
    path1 = storage.save_image("img_1", sample_image_data, collection="papers")
    path2 = storage.save_image("img_2", sample_image_data, collection="books")

    # 验证路径包含集合名称
    assert "papers" in path1
    assert "books" in path2

    # 验证文件存在于不同目录
    assert Path(path1).exists()
    assert Path(path2).exists()

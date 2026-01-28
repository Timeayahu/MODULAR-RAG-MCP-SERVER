"""
端到端测试：数据摄取流程 (Data Ingestion E2E Tests)

测试 scripts/ingest.py 的文件收集和命令行功能。
"""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """返回项目根目录。"""
    return Path(__file__).parent.parent.parent


@pytest.mark.e2e
def test_ingest_script_collect_single_file(tmp_path, project_root):
    """测试收集单个文件。"""
    sys.path.insert(0, str(project_root / "scripts"))
    from ingest import collect_files

    # 创建测试文件
    test_file = tmp_path / "doc.pdf"
    test_file.touch()

    # 测试文件收集
    files = collect_files(test_file)
    assert len(files) == 1
    assert files[0] == test_file


@pytest.mark.e2e
def test_ingest_script_collect_directory(tmp_path, project_root):
    """测试收集目录下的所有 PDF 文件。"""
    sys.path.insert(0, str(project_root / "scripts"))
    from ingest import collect_files

    # 创建测试目录和文件
    test_dir = tmp_path / "documents"
    test_dir.mkdir()
    pdf1 = test_dir / "doc1.pdf"
    pdf2 = test_dir / "doc2.pdf"
    txt_file = test_dir / "readme.txt"

    pdf1.touch()
    pdf2.touch()
    txt_file.touch()

    # 测试收集 PDF 文件
    files = collect_files(test_dir)
    assert len(files) == 2
    assert pdf1 in files
    assert pdf2 in files
    assert txt_file not in files


@pytest.mark.e2e
def test_ingest_script_nonexistent_path(tmp_path, project_root):
    """测试处理不存在的路径。"""
    sys.path.insert(0, str(project_root / "scripts"))
    from ingest import collect_files

    nonexistent = tmp_path / "nonexistent.pdf"
    with pytest.raises(FileNotFoundError):
        collect_files(nonexistent)


@pytest.mark.e2e
def test_ingest_script_empty_directory(tmp_path, project_root):
    """测试处理空目录（无 PDF 文件）。"""
    sys.path.insert(0, str(project_root / "scripts"))
    from ingest import collect_files

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    files = collect_files(empty_dir)
    assert len(files) == 0


@pytest.mark.e2e
def test_ingest_script_recursive_collection(tmp_path, project_root):
    """测试递归收集子目录中的 PDF 文件。"""
    sys.path.insert(0, str(project_root / "scripts"))
    from ingest import collect_files

    # 创建嵌套目录结构
    root = tmp_path / "root"
    subdir1 = root / "subdir1"
    subdir2 = root / "subdir2"
    subdir1.mkdir(parents=True)
    subdir2.mkdir(parents=True)

    pdf1 = root / "doc1.pdf"
    pdf2 = subdir1 / "doc2.pdf"
    pdf3 = subdir2 / "doc3.pdf"

    pdf1.touch()
    pdf2.touch()
    pdf3.touch()

    # 测试递归收集
    files = collect_files(root)
    assert len(files) == 3
    assert pdf1 in files
    assert pdf2 in files
    assert pdf3 in files

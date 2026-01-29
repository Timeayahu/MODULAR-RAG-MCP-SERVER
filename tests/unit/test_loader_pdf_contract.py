"""PDF Loader 契约测试。

验证要点：
- `PdfLoader.load(path)` 能返回 `Document`；
- `metadata` 至少包含 `source_path`；
- `doc_type` 为 "pdf"；
- 文本内容与样例文件一致（在当前阶段为伪 PDF 文本）。
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.models import Document  # noqa: E402
from libs.loader.pdf_loader import PdfLoader  # noqa: E402
from libs.loader.base_loader import BaseLoader  # noqa: E402


@pytest.mark.unit
def test_pdf_loader_returns_document_and_metadata(tmp_path: Path):
    # 使用临时目录中的伪 PDF 文件，避免对真实二进制 PDF 的依赖
    sample_pdf = tmp_path / "sample.pdf"
    content = "This is a fake PDF content for testing PdfLoader."
    sample_pdf.write_text(content, encoding="utf-8")

    loader: BaseLoader = PdfLoader()
    doc = loader.load(str(sample_pdf))

    assert isinstance(doc, Document)
    assert doc.text == content

    # metadata 契约字段
    assert BaseLoader.META_SOURCE_PATH in doc.metadata
    assert (
        Path(doc.metadata[BaseLoader.META_SOURCE_PATH]).resolve()
        == sample_pdf.resolve()
    )
    assert doc.metadata[BaseLoader.META_DOC_TYPE] == PdfLoader.DOC_TYPE


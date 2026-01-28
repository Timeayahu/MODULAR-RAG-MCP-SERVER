"""ImageCaptioner 测试 - 重点测试降级行为。

根据 DEV_SPEC C7 验收标准：
- 启用模式：存在 image_refs 时会生成 caption 并写入 metadata
- 降级模式：当配置禁用或异常时，chunk 保留 image_refs，标记 has_unprocessed_images
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from ingestion.models import Chunk
from ingestion.transform.image_captioner import ImageCaptioner


@pytest.mark.unit
def test_image_captioner_disabled_skips_processing():
    """ImageCaptioner 禁用时应跳过处理。"""
    captioner = ImageCaptioner(enabled=False)

    chunks = [
        Chunk(
            id="chunk-1",
            text="text with image",
            metadata={},
            image_refs=["img_1"],
        )
    ]

    result = captioner.transform(chunks)

    # 应返回原始 chunks，不做任何修改
    assert len(result) == 1
    assert result[0].text == "text with image"
    assert "image_captions" not in result[0].metadata


@pytest.mark.unit
def test_image_captioner_no_vision_llm_marks_unprocessed():
    """Vision LLM 未配置时应标记为未处理。"""
    captioner = ImageCaptioner(enabled=True, vision_llm=None)

    chunks = [
        Chunk(
            id="chunk-1",
            text="text with image",
            metadata={"source": "test.pdf"},
            image_refs=["img_1"],
        )
    ]

    result = captioner.transform(chunks)

    # 应标记为未处理
    assert len(result) == 1
    assert result[0].metadata.get("has_unprocessed_images") is True
    assert result[0].image_refs == ["img_1"]


@pytest.mark.unit
def test_image_captioner_generates_captions_when_enabled():
    """启用且配置正确时应生成图片描述。"""
    # Mock Vision LLM
    mock_llm = Mock()
    mock_llm.chat.return_value = "This image shows a diagram of system architecture."

    captioner = ImageCaptioner(
        enabled=True,
        vision_llm=mock_llm,
        inject_to_text=True,
    )

    chunks = [
        Chunk(
            id="chunk-1",
            text="See the architecture diagram below.",
            metadata={"source": "design.pdf", "page": 5},
            image_refs=["img_1"],
        )
    ]

    result = captioner.transform(chunks)

    # 验证调用了 Vision LLM
    assert mock_llm.chat.called

    # 验证生成了描述
    assert len(result) == 1
    assert "image_captions" in result[0].metadata
    assert len(result[0].metadata["image_captions"]) == 1
    assert result[0].metadata["image_captions"][0]["image_id"] == "img_1"
    assert "architecture" in result[0].metadata["image_captions"][0]["caption"]

    # 验证描述被注入到文本中
    assert "[图片描述:" in result[0].text


@pytest.mark.unit
def test_image_captioner_skips_chunks_without_images():
    """没有图片引用的 Chunk 应被跳过。"""
    mock_llm = Mock()
    captioner = ImageCaptioner(enabled=True, vision_llm=mock_llm)

    chunks = [
        Chunk(
            id="chunk-1",
            text="plain text chunk",
            metadata={},
            image_refs=[],
        )
    ]

    result = captioner.transform(chunks)

    # Vision LLM 不应被调用
    assert not mock_llm.chat.called

    # Chunk 应保持不变
    assert len(result) == 1
    assert result[0].text == "plain text chunk"
    assert "image_captions" not in result[0].metadata


@pytest.mark.unit
def test_image_captioner_handles_llm_error_gracefully():
    """Vision LLM 调用失败时应降级处理。"""
    # Mock Vision LLM 抛出异常
    mock_llm = Mock()
    mock_llm.chat.side_effect = RuntimeError("API Error")

    captioner = ImageCaptioner(enabled=True, vision_llm=mock_llm)

    chunks = [
        Chunk(
            id="chunk-1",
            text="text with image",
            metadata={},
            image_refs=["img_1"],
        )
    ]

    result = captioner.transform(chunks)

    # 应标记为未处理，不抛出异常
    assert len(result) == 1
    assert result[0].metadata.get("has_unprocessed_images") is True
    assert result[0].image_refs == ["img_1"]


@pytest.mark.unit
def test_image_captioner_processes_multiple_images():
    """应正确处理包含多个图片的 Chunk。"""
    mock_llm = Mock()
    mock_llm.chat.side_effect = [
        "Description for image 1",
        "Description for image 2",
    ]

    captioner = ImageCaptioner(enabled=True, vision_llm=mock_llm)

    chunks = [
        Chunk(
            id="chunk-1",
            text="text with two images",
            metadata={},
            image_refs=["img_1", "img_2"],
        )
    ]

    result = captioner.transform(chunks)

    # 应生成两个描述
    assert len(result) == 1
    assert len(result[0].metadata["image_captions"]) == 2
    assert result[0].metadata["image_captions"][0]["image_id"] == "img_1"
    assert result[0].metadata["image_captions"][1]["image_id"] == "img_2"


@pytest.mark.unit
def test_image_captioner_caching():
    """应缓存相同图片的描述，避免重复调用。"""
    mock_llm = Mock()
    mock_llm.chat.return_value = "Cached description"

    captioner = ImageCaptioner(enabled=True, vision_llm=mock_llm)

    chunks = [
        Chunk(
            id="chunk-1",
            text="first chunk",
            metadata={},
            image_refs=["img_1"],
        ),
        Chunk(
            id="chunk-2",
            text="second chunk",
            metadata={},
            image_refs=["img_1"],  # 相同的图片
        ),
    ]

    result = captioner.transform(chunks)

    # Vision LLM 应只被调用一次
    assert mock_llm.chat.call_count == 1

    # 两个 chunk 都应有描述
    assert len(result) == 2
    assert "image_captions" in result[0].metadata
    assert "image_captions" in result[1].metadata


@pytest.mark.unit
def test_image_captioner_inject_to_text_flag():
    """inject_to_text=False 时不应修改 chunk.text。"""
    mock_llm = Mock()
    mock_llm.chat.return_value = "Image description"

    captioner = ImageCaptioner(
        enabled=True,
        vision_llm=mock_llm,
        inject_to_text=False,  # 不注入文本
    )

    original_text = "original text"
    chunks = [
        Chunk(
            id="chunk-1",
            text=original_text,
            metadata={},
            image_refs=["img_1"],
        )
    ]

    result = captioner.transform(chunks)

    # 文本不应被修改
    assert result[0].text == original_text
    # 但 metadata 应包含描述
    assert "image_captions" in result[0].metadata

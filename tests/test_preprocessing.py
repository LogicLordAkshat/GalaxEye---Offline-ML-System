import io
import pytest
from PIL import Image
import torch
from app.services.preprocessing_service import (
    preprocessing_service,
    InvalidImageError,
    CorruptImageError
)

def create_sample_png_bytes(size=(64, 64), mode="RGB", color=(100, 150, 200)) -> bytes:
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_valid_image_preprocessing():
    img_bytes = create_sample_png_bytes(size=(64, 64), mode="RGB")
    tensor, tile_hash = preprocessing_service.validate_and_preprocess(img_bytes, "test_tile.png")
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    assert len(tile_hash) == 64  # SHA-256 hex length

def test_deterministic_tile_hashing():
    img_bytes = create_sample_png_bytes(size=(64, 64), mode="RGB")
    hash1 = preprocessing_service.compute_tile_hash(img_bytes)
    hash2 = preprocessing_service.compute_tile_hash(img_bytes)
    assert hash1 == hash2

def test_rgba_to_rgb_conversion():
    rgba_bytes = create_sample_png_bytes(size=(64, 64), mode="RGBA", color=(100, 150, 200, 255))
    tensor, _ = preprocessing_service.validate_and_preprocess(rgba_bytes, "test_rgba.png")
    assert tensor.shape == (1, 3, 64, 64)

def test_non_standard_dimension_resizing():
    large_bytes = create_sample_png_bytes(size=(128, 128), mode="RGB")
    tensor, _ = preprocessing_service.validate_and_preprocess(large_bytes, "test_large.png")
    assert tensor.shape == (1, 3, 64, 64)

def test_empty_image_rejection():
    with pytest.raises(InvalidImageError) as exc_info:
        preprocessing_service.validate_and_preprocess(b"", "empty.png")
    assert "empty" in str(exc_info.value)
    assert exc_info.value.stage == "validation"

def test_corrupt_image_rejection():
    corrupt_bytes = b"This is not a real image binary file data"
    with pytest.raises(CorruptImageError) as exc_info:
        preprocessing_service.validate_and_preprocess(corrupt_bytes, "corrupt.png")
    assert "corrupt" in str(exc_info.value).lower()
    assert exc_info.value.stage == "decode"

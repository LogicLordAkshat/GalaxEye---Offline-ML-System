import io
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from app.core.config import settings
from app.core.logging import logger

class PreprocessingError(Exception):
    """Base exception for image validation and preprocessing errors."""
    def __init__(self, message: str, stage: str = "preprocessing"):
        super().__init__(message)
        self.message = message
        self.stage = stage

class InvalidImageError(PreprocessingError):
    def __init__(self, message: str):
        super().__init__(message, stage="validation")

class CorruptImageError(PreprocessingError):
    def __init__(self, message: str):
        super().__init__(message, stage="decode")

class PreprocessingService:
    """
    Image validation and preprocessing pipeline for satellite tiles.
    Produces deterministic byte hashes and normalized PyTorch tensors.
    """
    def __init__(self):
        self.target_size = settings.TARGET_IMAGE_SIZE
        self.max_size_bytes = settings.MAX_IMAGE_SIZE_BYTES

    def compute_tile_hash(self, file_bytes: bytes) -> str:
        """
        Computes SHA-256 hash of the exact input bytes for tile provenance and duplicate detection.
        """
        return hashlib.sha256(file_bytes).hexdigest()

    def validate_and_preprocess(self, file_bytes: bytes, filename: str) -> tuple[torch.Tensor, str]:
        """
        Validates the raw image payload, decodes it safely, computes hash, and returns
        the normalized tensor (1, 3, H, W) along with the tile hash.
        """
        # 1. Byte length and filename checks
        if not file_bytes:
            raise InvalidImageError("Uploaded file is empty (0 bytes).")

        suffix = Path(filename).suffix.lower()
        if suffix not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise InvalidImageError(
                f"Unsupported image extension '{suffix or '<none>'}'. "
                f"Allowed extensions: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}."
            )
            
        if len(file_bytes) > self.max_size_bytes:
            raise InvalidImageError(f"Image size ({len(file_bytes)} bytes) exceeds maximum limit of {self.max_size_bytes} bytes.")

        # 2. Compute deterministic tile hash
        tile_hash = self.compute_tile_hash(file_bytes)

        # 3. Decode image with PIL
        try:
            image_stream = io.BytesIO(file_bytes)
            img = Image.open(image_stream)
            img.verify()  # Verify image integrity without decoding pixel data yet
        except (UnidentifiedImageError, SyntaxError, ValueError) as e:
            raise CorruptImageError(f"File '{filename}' is corrupt or not a recognized image format.") from e
        except Exception as e:
            raise CorruptImageError(f"Failed to parse image data: {str(e)}") from e

        # 4. Re-open for actual reading (verify() closes/invalidates the stream)
        try:
            image_stream.seek(0)
            img = Image.open(image_stream)
            
            # Convert to RGB (handles RGBA, Greyscale, Paletted, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Validate or resize to target resolution (64x64)
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.BILINEAR)

            # Convert to float32 array normalized to [0, 1]
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            # Reorder dimensions from (H, W, C) to (C, H, W)
            tensor_arr = img_arr.transpose(2, 0, 1)
            
            # Add batch dimension -> (1, 3, 64, 64)
            tensor = torch.tensor(tensor_arr, dtype=torch.float32).unsqueeze(0)
            
            return tensor, tile_hash
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise CorruptImageError(f"Error during image tensor transformation: {str(e)}") from e

preprocessing_service = PreprocessingService()

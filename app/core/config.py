import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application Configuration loaded from environment variables or sensible defaults.
    Designed for 100% offline standalone execution.
    """
    APP_NAME: str = "GalaxEye Offline Satellite Tile Classification Service"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Host
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model Artifacts
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent.parent / "models" / "landuse_cnn_v1.pt"))
    METADATA_PATH: str = os.getenv("METADATA_PATH", str(Path(__file__).resolve().parent.parent.parent / "models" / "model_metadata.json"))
    MODEL_NAME: str = "landuse_cnn"
    # Matches the packaged artifact and model_metadata.json.
    MODEL_VERSION: str = "1.0.0"
    PREPROCESSING_VERSION: str = "1.0.0"
    
    # Classification / Confidence Policy
    # Operational design choice for ACCEPTED vs UNCERTAIN classification;
    # not a statistically calibrated probability threshold.
    CONFIDENCE_THRESHOLD: float = 0.70
    
    # Database Configuration (SQLite default with WAL mode)
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent.parent.parent / "data" / "predictions.db"))
    
    # Validation constraints
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]
    TARGET_IMAGE_SIZE: tuple[int, int] = (64, 64)
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # 'json' or 'text'

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")

settings = Settings()

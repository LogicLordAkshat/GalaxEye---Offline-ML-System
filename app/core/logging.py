import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any
from app.core.config import settings

class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for offline observability.
    Enables post-deployment automated log aggregation and parsing without live operators.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Include custom structured attributes if provided
        for key in ("event", "tile_hash", "prediction_id", "status", "predicted_class", 
                    "confidence", "latency_ms", "model_version", "error", "client_ip", "path"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
                
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("galaxeye")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    
    # Avoid duplicate handlers if reloaded
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if settings.LOG_FORMAT.lower() == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

logger = setup_logging()

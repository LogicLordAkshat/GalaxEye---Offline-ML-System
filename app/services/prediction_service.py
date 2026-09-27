import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.db.models import PredictionRecord
from app.ml.inference import inference_engine
from app.ml.model import model_manager
from app.repositories.prediction_repository import prediction_repository
from app.schemas.prediction import PredictionResponse
from app.services.preprocessing_service import preprocessing_service

class PredictionService:
    """
    Core Domain Service orchestrating validation, inference, triage prioritization, and persistence.
    """
    def __init__(self):
        self.repository = prediction_repository
        self.preprocessor = preprocessing_service
        self.engine = inference_engine

    def calculate_review_priority(
        self,
        confidence: float,
        confidence_margin: float
    ) -> tuple[float, str, str]:
        """
        Operational triage heuristic combining model uncertainty and top-2 ambiguity:
        - uncertainty = 1.0 - confidence
        - ambiguity   = 1.0 - confidence_margin
        - priority    = 0.5 * uncertainty + 0.5 * ambiguity
        
        Returns: (review_priority_score, priority_level, review_reason)
        """
        uncertainty = 1.0 - confidence
        ambiguity = 1.0 - confidence_margin
        priority_score = round(0.5 * uncertainty + 0.5 * ambiguity, 4)

        # Categorize into triage levels
        if priority_score >= 0.70:
            level = "CRITICAL"
        elif priority_score >= 0.50:
            level = "HIGH"
        elif priority_score >= 0.30:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Determine human-interpretable review reason
        is_low_conf = confidence < settings.CONFIDENCE_THRESHOLD
        is_small_margin = confidence_margin <= 0.15

        if is_low_conf and is_small_margin:
            reason = "LOW_CONFIDENCE_AND_SMALL_MARGIN"
        elif is_low_conf:
            reason = "LOW_CONFIDENCE"
        elif is_small_margin:
            reason = "SMALL_TOP2_MARGIN"
        else:
            reason = "STANDARD_CONFIDENCE"

        return priority_score, level, reason

    def predict_tile(
        self,
        file_bytes: bytes,
        filename: str,
        force_recompute: bool = False
    ) -> PredictionResponse:
        """
        Executes end-to-end prediction pipeline for a single satellite tile image.
        """
        t0 = time.perf_counter()
        
        # 1. Image validation & preprocessing
        tensor, tile_hash = self.preprocessor.validate_and_preprocess(file_bytes, filename)
        
        # 2. Duplicate detection / Idempotency check (respects current model version & checksum)
        if not force_recompute:
            cached_record = self.repository.get_by_tile_hash(tile_hash)
            if cached_record and cached_record.model_version == settings.MODEL_VERSION and cached_record.model_checksum == model_manager.checksum:
                logger.info(
                    f"Duplicate tile detected (hash: {tile_hash[:12]}...). Returning cached prediction.",
                    extra={
                        "event": "prediction_cache_hit",
                        "tile_hash": tile_hash,
                        "prediction_id": cached_record.prediction_id,
                        "predicted_class": cached_record.predicted_class,
                        "status": cached_record.status,
                        "review_priority": cached_record.review_priority
                    }
                )
                return self._record_to_response(cached_record, is_cached=True)

        # 3. Local Model Forward Inference on CPU
        inf_result = self.engine.predict(tensor)
        predicted_class = inf_result["predicted_class"]
        confidence = inf_result["confidence"]
        top_2_class = inf_result["top_2_class"]
        top_2_confidence = inf_result["top_2_confidence"]
        confidence_margin = inf_result["confidence_margin"]
        probabilities = inf_result["probabilities"]
        inference_latency_ms = inf_result["inference_latency_ms"]

        # 4. Confidence Policy Decision
        status = "ACCEPTED" if confidence >= settings.CONFIDENCE_THRESHOLD else "UNCERTAIN"

        # 5. Analyst Review Priority Engine
        review_priority, priority_level, review_reason = self.calculate_review_priority(
            confidence=confidence,
            confidence_margin=confidence_margin
        )

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        prediction_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # 6. Build record with complete provenance and triage metadata
        record = PredictionRecord(
            prediction_id=prediction_id,
            tile_hash=tile_hash,
            filename=filename,
            predicted_class=predicted_class,
            confidence=confidence,
            top_2_class=top_2_class,
            top_2_confidence=top_2_confidence,
            confidence_margin=confidence_margin,
            review_priority=review_priority,
            review_priority_level=priority_level,
            review_reason=review_reason,
            status=status,
            all_probabilities=json.dumps(probabilities),
            model_name=settings.MODEL_NAME,
            model_version=settings.MODEL_VERSION,
            model_checksum=model_manager.checksum,
            preprocessing_version=settings.PREPROCESSING_VERSION,
            inference_latency_ms=inference_latency_ms,
            total_latency_ms=total_latency_ms,
            created_at=created_at
        )

        # 7. Persist to database
        self.repository.save(record)

        logger.info(
            f"Prediction completed: {predicted_class} ({confidence:.2f}, {status}) [Priority: {priority_level} {review_priority:.2f}] in {total_latency_ms}ms",
            extra={
                "event": "prediction_success",
                "tile_hash": tile_hash,
                "prediction_id": prediction_id,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "status": status,
                "review_priority": review_priority,
                "review_priority_level": priority_level,
                "latency_ms": total_latency_ms,
                "model_version": settings.MODEL_VERSION
            }
        )

        return self._record_to_response(record, is_cached=False)

    def _record_to_response(self, record: PredictionRecord, is_cached: bool = False) -> PredictionResponse:
        return PredictionResponse(
            prediction_id=record.prediction_id,
            tile_hash=record.tile_hash,
            filename=record.filename,
            predicted_class=record.predicted_class,
            confidence=record.confidence,
            top_2_class=record.top_2_class,
            top_2_confidence=record.top_2_confidence,
            confidence_margin=record.confidence_margin,
            review_priority=record.review_priority,
            review_priority_level=record.review_priority_level,
            review_reason=record.review_reason,
            status=record.status,
            probabilities=json.loads(record.all_probabilities),
            model_name=record.model_name,
            model_version=record.model_version,
            model_checksum=record.model_checksum,
            preprocessing_version=record.preprocessing_version,
            inference_latency_ms=record.inference_latency_ms,
            total_latency_ms=record.total_latency_ms,
            created_at=record.created_at,
            is_cached=is_cached
        )

prediction_service = PredictionService()

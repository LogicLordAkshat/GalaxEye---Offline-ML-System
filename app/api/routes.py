from typing import List, Optional
import io
import csv
from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status, Response
from app.core.config import settings
from app.core.logging import logger
from app.db.database import db
from app.db.models import ReviewRecord
from app.ml.model import model_manager
from app.repositories.prediction_repository import prediction_repository
from app.schemas.prediction import (
    PredictionResponse,
    PredictionListResponse,
    PredictionSummaryResponse,
    ShortlistResponse,
    HumanReviewRequest,
    HumanReviewResponse,
    HealthResponse
)
from app.services.prediction_service import prediction_service
from app.services.preprocessing_service import PreprocessingError, InvalidImageError, CorruptImageError

router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a satellite image tile",
    description="Accepts a satellite image tile, validates it, runs local offline inference, calculates runner-up competition, assigns review priority, and persists results."
)
async def predict_tile(
    file: UploadFile = File(..., description="Satellite image tile (PNG, JPG, TIFF, etc.)"),
    force_recompute: bool = Query(False, description="If True, bypass duplicate cache and re-run model inference")
):
    try:
        content = await file.read()
        filename = file.filename or "unknown_tile.png"
        
        response = prediction_service.predict_tile(
            file_bytes=content,
            filename=filename,
            force_recompute=force_recompute
        )
        return response
        
    except InvalidImageError as e:
        logger.warning(f"Validation error on upload '{file.filename}': {e.message}", extra={"event": "validation_error", "stage": e.stage})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"stage": e.stage, "message": e.message}
        )
    except CorruptImageError as e:
        logger.warning(f"Corrupt image on upload '{file.filename}': {e.message}", extra={"event": "corrupt_image_error", "stage": e.stage})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"stage": e.stage, "message": e.message}
        )
    except PreprocessingError as e:
        logger.warning(f"Preprocessing error on upload '{file.filename}': {e.message}", extra={"event": "preprocessing_error", "stage": e.stage})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"stage": e.stage, "message": e.message}
        )
    except Exception as e:
        logger.error(f"Internal error processing tile '{file.filename}': {str(e)}", exc_info=True, extra={"event": "internal_error"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"stage": "inference_or_persistence", "message": "Failed to process image tile due to an internal error."}
        )

@router.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    status_code=status.HTTP_200_OK,
    summary="Classify a batch of satellite image tiles",
    description="Convenience endpoint to upload and process multiple satellite image tiles in a single request."
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of satellite image tiles"),
    force_recompute: bool = Query(False, description="If True, bypass duplicate cache")
):
    results = []
    for file in files:
        content = await file.read()
        filename = file.filename or "unknown_tile.png"
        try:
            res = prediction_service.predict_tile(content, filename, force_recompute=force_recompute)
            results.append(res)
        except Exception as e:
            logger.warning(f"Error in batch item '{filename}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"stage": "batch_item_error", "filename": filename, "message": str(e)}
            )
    return results

@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    summary="Query stored predictions",
    description="Query and filter historical predictions by land-use class, confidence status, review priority, or model version."
)
async def get_predictions(
    class_name: Optional[str] = Query(None, description="Filter by predicted land-use class (e.g. Forest, Highway, Industrial)"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: 'ACCEPTED' or 'UNCERTAIN'"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score filter [0.0 - 1.0]"),
    min_priority: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum review priority filter [0.0 - 1.0]"),
    priority_level: Optional[str] = Query(None, description="Filter by triage level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'"),
    model_version: Optional[str] = Query(None, description="Filter by model version (e.g. 1.0.0)"),
    sort_by: Optional[str] = Query("id", description="Sort by: 'id', 'review_priority', 'confidence', 'confidence_margin', 'created_at'"),
    sort_order: Optional[str] = Query("desc", description="Sort order: 'asc' or 'desc'"),
    limit: int = Query(50, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Records offset for pagination")
):
    records, total = prediction_repository.query(
        class_name=class_name,
        status=status_filter,
        min_confidence=min_confidence,
        min_priority=min_priority,
        priority_level=priority_level,
        model_version=model_version,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset
    )
    items = [prediction_service._record_to_response(r) for r in records]
    return PredictionListResponse(
        total=total,
        count=len(items),
        limit=limit,
        offset=offset,
        items=items
    )

@router.get(
    "/predictions/shortlist",
    response_model=ShortlistResponse,
    summary="Analyst Triage & Shortlisting",
    description=(
        "Intelligently shortlist tiles using one of 5 triage rules:\n"
        "- **needs_review**: Uncertain tiles or high review-priority tiles.\n"
        "- **high_confidence**: Auto-approved tiles (confidence ≥ 85%).\n"
        "- **conflict_margin**: Competing top-2 classes (margin ≤ max_margin).\n"
        "- **critical_priority**: CRITICAL triage priority tiles.\n"
        "- **per_class_anomaly**: Statistical outliers — tiles whose confidence is >1.5σ below their class mean, "
        "surfacing likely mis-classified or ambiguous samples.\n\n"
        "Optional: filter by class, date range, confidence bounds."
    )
)
async def shortlist_predictions(
    mode: str = Query("needs_review", description="Shortlist mode: 'needs_review' | 'high_confidence' | 'conflict_margin' | 'critical_priority' | 'per_class_anomaly'"),
    class_name: Optional[str] = Query(None, description="Filter by specific predicted class (optional)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Min confidence boundary"),
    max_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Max confidence boundary"),
    min_priority: Optional[float] = Query(None, ge=0.0, le=1.0, description="Min review priority score"),
    max_margin: float = Query(0.15, ge=0.0, le=1.0, description="Max probability margin between top 2 classes"),
    date_from: Optional[str] = Query(None, description="ISO 8601 start timestamp filter (e.g. 2026-09-25T00:00:00Z)"),
    date_to: Optional[str] = Query(None, description="ISO 8601 end timestamp filter (e.g. 2026-09-30T23:59:59Z)"),
    limit: int = Query(50, ge=1, le=500, description="Max shortlisted records to return")
):
    records, total, rule_desc = prediction_repository.shortlist(
        mode=mode,
        class_name=class_name,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        min_priority=min_priority,
        max_margin=max_margin,
        date_from=date_from,
        date_to=date_to,
        limit=limit
    )
    items = [prediction_service._record_to_response(r) for r in records]
    return ShortlistResponse(
        total=total,
        criteria=rule_desc,
        count=len(items),
        items=items
    )

@router.post(
    "/predictions/{prediction_id}/review",
    response_model=HumanReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit human analyst review feedback",
    description="Captures human-in-the-loop analyst decisions (ACCEPT, REJECT, CORRECT_CLASS) for audit trails and future offline dataset curation."
)
async def submit_prediction_review(
    prediction_id: str,
    payload: HumanReviewRequest
):
    # Check that prediction exists
    record = prediction_repository.get_by_id(prediction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"stage": "review", "message": f"Prediction ID '{prediction_id}' does not exist."}
        )

    valid_decisions = {"ACCEPT", "REJECT", "CORRECT_CLASS"}
    decision = payload.decision.upper()
    if decision not in valid_decisions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"stage": "review", "message": f"Decision must be one of: {', '.join(sorted(valid_decisions))}."}
        )
    if decision == "CORRECT_CLASS":
        if payload.reviewed_class not in model_manager.classes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"stage": "review", "message": "reviewed_class must be a valid model class for CORRECT_CLASS."}
            )
    elif payload.reviewed_class is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"stage": "review", "message": "reviewed_class is only allowed for CORRECT_CLASS."}
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()
    review = ReviewRecord(
        prediction_id=prediction_id,
        decision=decision,
        reviewed_class=payload.reviewed_class,
        comment=payload.comment,
        reviewer_id=payload.reviewer_id or "analyst_offline",
        reviewed_at=reviewed_at
    )
    saved_review = prediction_repository.save_review(review)

    logger.info(
        f"Analyst review submitted for {prediction_id}: {payload.decision}",
        extra={
            "event": "analyst_review_submitted",
            "prediction_id": prediction_id,
            "decision": decision,
            "reviewed_class": payload.reviewed_class
        }
    )

    return HumanReviewResponse(
        review_id=saved_review.id or 1,
        prediction_id=saved_review.prediction_id,
        decision=saved_review.decision,
        reviewed_class=saved_review.reviewed_class,
        comment=saved_review.comment,
        reviewer_id=saved_review.reviewer_id or "analyst_offline",
        reviewed_at=saved_review.reviewed_at
    )

@router.get(
    "/predictions/{prediction_id}/reviews",
    response_model=List[HumanReviewResponse],
    summary="Get human review history for a prediction",
    description="Retrieves the human review audit log for a specific prediction ID."
)
async def get_prediction_reviews(prediction_id: str):
    reviews = prediction_repository.get_reviews_for_prediction(prediction_id)
    return [
        HumanReviewResponse(
            review_id=r.id or 0,
            prediction_id=r.prediction_id,
            decision=r.decision,
            reviewed_class=r.reviewed_class,
            comment=r.comment,
            reviewer_id=r.reviewer_id or "analyst_offline",
            reviewed_at=r.reviewed_at
        )
        for r in reviews
    ]

@router.get(
    "/predictions/export/csv",
    summary="Export predictions as CSV",
    description="Download a full CSV export of stored predictions including triage priority and margin metrics for spreadsheet / GIS analysis."
)
async def export_predictions_csv(
    limit: int = Query(1000, ge=1, le=10000, description="Max records to export")
):
    records, _ = prediction_repository.query(limit=limit, offset=0)
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "prediction_id", "tile_hash", "filename", "predicted_class",
        "confidence", "top_2_class", "top_2_confidence", "confidence_margin",
        "review_priority", "review_priority_level", "review_reason",
        "status", "model_version", "model_checksum",
        "inference_latency_ms", "total_latency_ms", "created_at"
    ])
    
    for r in records:
        writer.writerow([
            r.prediction_id, r.tile_hash, r.filename, r.predicted_class,
            f"{r.confidence:.4f}", r.top_2_class, f"{r.top_2_confidence:.4f}", f"{r.confidence_margin:.4f}",
            f"{r.review_priority:.4f}", r.review_priority_level, r.review_reason,
            r.status, r.model_version, r.model_checksum,
            f"{r.inference_latency_ms:.2f}", f"{r.total_latency_ms:.2f}", r.created_at
        ])
        
    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=galaxeye_predictions_export.csv"}
    )

@router.get(
    "/predictions/summary",
    response_model=PredictionSummaryResponse,
    summary="Observability summary of predictions",
    description="Provides aggregate statistics on class distribution, certainty rates, review priority levels, average confidence, and latency."
)
async def get_prediction_summary():
    stats = prediction_repository.get_summary_statistics()
    return PredictionSummaryResponse(
        total_predictions=stats["total_predictions"],
        status_distribution=stats["status_distribution"],
        priority_distribution=stats["priority_distribution"],
        class_distribution=stats["class_distribution"],
        average_confidence=stats["average_confidence"],
        average_review_priority=stats["average_review_priority"],
        average_inference_latency_ms=stats["average_inference_latency_ms"],
        model_version=settings.MODEL_VERSION,
        model_checksum=model_manager.checksum
    )

@router.get(
    "/predictions/{id_or_hash}",
    response_model=PredictionResponse,
    summary="Retrieve prediction by UUID or tile hash",
    description="Fetch a specific prediction record using either its UUID prediction_id or SHA-256 tile_hash."
)
async def get_prediction_by_id_or_hash(id_or_hash: str):
    record = prediction_repository.get_by_id(id_or_hash)
    if not record:
        record = prediction_repository.get_by_tile_hash(id_or_hash)
        
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"stage": "retrieval", "message": f"Prediction with ID or Hash '{id_or_hash}' not found."}
        )
    return prediction_service._record_to_response(record)

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Verifies local database connectivity and model status without external network calls."
)
async def health_check():
    db_ok = True
    try:
        with db.session() as conn:
            conn.execute("SELECT 1;").fetchone()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_ok = False

    model_ok = model_manager.is_loaded and model_manager.model is not None
    overall_status = "healthy" if (db_ok and model_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        service_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database_status="connected" if db_ok else "unreachable",
        model_loaded=model_ok,
        model_name=settings.MODEL_NAME,
        model_version=settings.MODEL_VERSION,
        model_checksum=model_manager.checksum,
        offline_mode=True
    )

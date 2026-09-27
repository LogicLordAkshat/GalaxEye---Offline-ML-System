from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """
    Schema for a single tile prediction result.
    Provides complete model provenance, runner-up competition, and triage priority metadata.
    """
    prediction_id: str = Field(..., description="Unique UUID for this prediction event")
    tile_hash: str = Field(..., description="Deterministic SHA-256 hash of tile image bytes")
    filename: str = Field(..., description="Original filename of the uploaded tile")
    predicted_class: str = Field(..., description="Top predicted land-use class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for top class [0.0 - 1.0]")
    top_2_class: str = Field(..., description="Runner-up predicted land-use class")
    top_2_confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax score for runner-up class")
    confidence_margin: float = Field(..., ge=0.0, le=1.0, description="Difference between top-1 and runner-up probability (top1 - top2)")
    review_priority: float = Field(..., ge=0.0, le=1.0, description="Analyst triage priority score [0.0 - 1.0] (higher = needs human attention first)")
    review_priority_level: str = Field(..., description="Triage category: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    review_reason: str = Field(..., description="Human-interpretable explanation for triage priority")
    status: str = Field(..., description="Prediction status: 'ACCEPTED' or 'UNCERTAIN' based on policy threshold")
    probabilities: Dict[str, float] = Field(..., description="Full class probability distribution")
    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model release version")
    model_checksum: str = Field(..., description="SHA-256 hash of the local model artifact for reproducibility")
    preprocessing_version: str = Field(..., description="Preprocessing pipeline version")
    inference_latency_ms: float = Field(..., description="Model forward pass latency in milliseconds")
    total_latency_ms: float = Field(..., description="Total end-to-end processing latency in milliseconds")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp")
    is_cached: bool = Field(False, description="True if result was returned from duplicate tile cache")

class PredictionQueryFilter(BaseModel):
    class_name: Optional[str] = Field(None, description="Filter by predicted land-use class")
    status: Optional[str] = Field(None, description="Filter by status ('ACCEPTED' or 'UNCERTAIN')")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter predictions with confidence >= min_confidence")
    min_priority: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter predictions with review_priority >= min_priority")
    priority_level: Optional[str] = Field(None, description="Filter by triage level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'")
    model_version: Optional[str] = Field(None, description="Filter by specific model version")
    sort_by: Optional[str] = Field("id", description="Sort field: 'id', 'review_priority', 'confidence', 'confidence_margin'")
    sort_order: Optional[str] = Field("desc", description="Sort order: 'asc' or 'desc'")
    limit: int = Field(50, ge=1, le=1000, description="Pagination limit (1-1000)")
    offset: int = Field(0, ge=0, description="Pagination offset")

class PredictionListResponse(BaseModel):
    total: int = Field(..., description="Total number of matching records")
    count: int = Field(..., description="Number of items in current page")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    items: List[PredictionResponse] = Field(..., description="List of prediction records")

class ClassDistribution(BaseModel):
    class_name: str
    count: int
    percentage: float

class StatusDistribution(BaseModel):
    status: str
    count: int
    percentage: float

class PriorityDistribution(BaseModel):
    level: str
    count: int
    percentage: float

class PredictionSummaryResponse(BaseModel):
    """
    Offline Observability Summary Schema for analysts and automated auditing.
    """
    total_predictions: int
    status_distribution: List[StatusDistribution]
    priority_distribution: List[PriorityDistribution]
    class_distribution: List[ClassDistribution]
    average_confidence: float
    average_review_priority: float
    average_inference_latency_ms: float
    model_version: str
    model_checksum: str

class ShortlistResponse(BaseModel):
    total: int = Field(..., description="Total records matching shortlist criteria")
    criteria: str = Field(..., description="Shortlist triage filter rule applied")
    count: int = Field(..., description="Returned items count")
    items: List[PredictionResponse] = Field(..., description="List of shortlisted predictions")

class HumanReviewRequest(BaseModel):
    decision: str = Field(..., description="Analyst decision: 'ACCEPT', 'REJECT', or 'CORRECT_CLASS'")
    reviewed_class: Optional[str] = Field(None, description="Corrected land-use class if decision is 'CORRECT_CLASS'")
    comment: Optional[str] = Field(None, description="Optional analyst notes on visual inspection")
    reviewer_id: Optional[str] = Field("analyst_offline", description="Identifier of the reviewing analyst")

class HumanReviewResponse(BaseModel):
    review_id: int = Field(..., description="Auto-incremented review ID")
    prediction_id: str = Field(..., description="Reviewed prediction UUID")
    decision: str = Field(..., description="Analyst decision")
    reviewed_class: Optional[str] = Field(None, description="Corrected class if provided")
    comment: Optional[str] = Field(None, description="Analyst comment")
    reviewer_id: str = Field(..., description="Reviewer identifier")
    reviewed_at: str = Field(..., description="ISO 8601 UTC review timestamp")

class HealthResponse(BaseModel):
    status: str
    service_name: str
    version: str
    database_status: str
    model_loaded: bool
    model_name: str
    model_version: str
    model_checksum: str
    offline_mode: bool = True

class ErrorDetail(BaseModel):
    stage: str
    message: str
    detail: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail

from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class PredictionRecord:
    """
    Data model representing a persisted satellite tile prediction record in SQLite.
    Stores complete provenance, top-2 class competition, uncertainty signals, and triage priority.
    """
    prediction_id: str
    tile_hash: str
    filename: str
    predicted_class: str
    confidence: float
    top_2_class: str
    top_2_confidence: float
    confidence_margin: float
    review_priority: float
    review_priority_level: str  # 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    review_reason: str          # 'LOW_CONFIDENCE' | 'SMALL_TOP2_MARGIN' | 'LOW_CONFIDENCE_AND_SMALL_MARGIN' | 'STANDARD_CONFIDENCE'
    status: str                 # 'ACCEPTED' | 'UNCERTAIN'
    all_probabilities: str      # JSON string
    model_name: str
    model_version: str
    model_checksum: str
    preprocessing_version: str
    inference_latency_ms: float
    total_latency_ms: float
    created_at: str
    id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class ReviewRecord:
    """
    Data model representing human analyst triage feedback on a specific prediction.
    Human-in-the-loop audit feedback for offline evaluation & future curation.
    """
    prediction_id: str
    decision: str              # 'ACCEPT' | 'REJECT' | 'CORRECT_CLASS'
    reviewed_class: Optional[str] = None
    comment: Optional[str] = None
    reviewer_id: Optional[str] = "analyst_offline"
    reviewed_at: str = ""
    id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

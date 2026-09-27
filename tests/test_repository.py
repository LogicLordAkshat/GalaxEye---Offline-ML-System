import json
import pytest
from app.db.database import Database
from app.db.models import PredictionRecord, ReviewRecord
from app.repositories.prediction_repository import PredictionRepository

@pytest.fixture
def in_memory_repo():
    test_db = Database(db_path=":memory:")
    test_db.init_db()
    repo = PredictionRepository(database=test_db)
    return repo

def sample_record(
    prediction_id="p1",
    tile_hash="h1",
    class_name="Forest",
    conf=0.85,
    top2_class="River",
    top2_conf=0.10,
    margin=0.75,
    priority=0.20,
    priority_level="LOW",
    reason="STANDARD_CONFIDENCE",
    status="ACCEPTED"
):
    return PredictionRecord(
        prediction_id=prediction_id,
        tile_hash=tile_hash,
        filename="tile_sample.png",
        predicted_class=class_name,
        confidence=conf,
        top_2_class=top2_class,
        top_2_confidence=top2_conf,
        confidence_margin=margin,
        review_priority=priority,
        review_priority_level=priority_level,
        review_reason=reason,
        status=status,
        all_probabilities=json.dumps({class_name: conf, top2_class: top2_conf}),
        model_name="landuse_cnn",
        model_version="1.0.0",
        model_checksum="dummy_checksum_123",
        preprocessing_version="1.0.0",
        inference_latency_ms=1.2,
        total_latency_ms=3.4,
        created_at="2026-09-26T00:00:00Z"
    )

def test_save_and_retrieve_by_id(in_memory_repo):
    rec = sample_record("test-uuid-1", "hash-123", "Highway", 0.92, top2_class="River", top2_conf=0.05, margin=0.87, priority=0.10, priority_level="LOW", status="ACCEPTED")
    saved = in_memory_repo.save(rec)
    assert saved.id is not None

    fetched = in_memory_repo.get_by_id("test-uuid-1")
    assert fetched is not None
    assert fetched.prediction_id == "test-uuid-1"
    assert fetched.predicted_class == "Highway"
    assert fetched.top_2_class == "River"
    assert fetched.confidence == 0.92
    assert fetched.confidence_margin == 0.87
    assert fetched.review_priority == 0.10
    assert fetched.review_priority_level == "LOW"
    assert fetched.status == "ACCEPTED"
    assert fetched.tile_hash == "hash-123"

def test_get_by_tile_hash(in_memory_repo):
    rec = sample_record("test-uuid-2", "hash-duplicate-xyz", "River", 0.65, status="UNCERTAIN")
    in_memory_repo.save(rec)

    fetched = in_memory_repo.get_by_tile_hash("hash-duplicate-xyz")
    assert fetched is not None
    assert fetched.prediction_id == "test-uuid-2"
    assert fetched.predicted_class == "River"

def test_query_filtering_and_sorting(in_memory_repo):
    in_memory_repo.save(sample_record("p1", "h1", "Forest", 0.90, priority=0.15, priority_level="LOW", status="ACCEPTED"))
    in_memory_repo.save(sample_record("p2", "h2", "Forest", 0.55, priority=0.65, priority_level="HIGH", status="UNCERTAIN"))
    in_memory_repo.save(sample_record("p3", "h3", "Highway", 0.95, priority=0.08, priority_level="LOW", status="ACCEPTED"))
    in_memory_repo.save(sample_record("p4", "h4", "River", 0.40, priority=0.85, priority_level="CRITICAL", status="UNCERTAIN"))

    # Filter by class
    forests, count = in_memory_repo.query(class_name="Forest")
    assert count == 2
    assert len(forests) == 2

    # Filter by priority level
    critical, count_crit = in_memory_repo.query(priority_level="CRITICAL")
    assert count_crit == 1
    assert critical[0].prediction_id == "p4"

    # All records use the currently packaged model version in API flows.
    # Repository fixtures intentionally exercise persistence independently.
    # Sort by review priority descending
    sorted_recs, total = in_memory_repo.query(sort_by="review_priority", sort_order="desc")
    assert total == 4
    assert sorted_recs[0].prediction_id == "p4"
    assert sorted_recs[0].review_priority == 0.85

def test_shortlist_modes(in_memory_repo):
    in_memory_repo.save(sample_record("p1", "h1", "Forest", 0.90, margin=0.80, priority=0.15, priority_level="LOW", status="ACCEPTED"))
    in_memory_repo.save(sample_record("p2", "h2", "River", 0.48, top2_class="Highway", top2_conf=0.42, margin=0.06, priority=0.76, priority_level="CRITICAL", status="UNCERTAIN"))

    # Test conflict margin shortlist
    conflict_tiles, total_c, desc_c = in_memory_repo.shortlist(mode="conflict_margin", max_margin=0.15)
    assert total_c == 1
    assert conflict_tiles[0].prediction_id == "p2"

    # Test critical priority shortlist
    crit_tiles, total_cr, _ = in_memory_repo.shortlist(mode="critical_priority")
    assert total_cr == 1
    assert crit_tiles[0].prediction_id == "p2"

def test_shortlist_per_class_anomaly(in_memory_repo):
    """
    Per-class anomaly shortlist mode: surfaces tiles whose confidence is >1.5σ
    below the mean confidence for their predicted class.

    Setup: 4 Forest tiles with high confidence (~0.90 mean) + 1 anomalous outlier
    at 0.20. Only the outlier should appear in the anomaly shortlist.
    """
    # 4 normal Forest tiles — class mean confidence ≈ 0.90
    in_memory_repo.save(sample_record("pa1", "ha1", "Forest", 0.92, status="ACCEPTED"))
    in_memory_repo.save(sample_record("pa2", "ha2", "Forest", 0.89, status="ACCEPTED"))
    in_memory_repo.save(sample_record("pa3", "ha3", "Forest", 0.91, status="ACCEPTED"))
    in_memory_repo.save(sample_record("pa4", "ha4", "Forest", 0.88, status="ACCEPTED"))
    # 1 anomalously low-confidence Forest tile — expected outlier
    in_memory_repo.save(sample_record("pa5", "ha5", "Forest", 0.20, priority=0.85, priority_level="CRITICAL", status="UNCERTAIN"))

    anomalies, total_a, desc_a = in_memory_repo.shortlist(mode="per_class_anomaly")
    assert total_a >= 1, "Expected at least one anomaly to be detected"
    ids = [r.prediction_id for r in anomalies]
    assert "pa5" in ids, "Low-confidence outlier tile must appear in per_class_anomaly shortlist"
    assert "pa1" not in ids, "High-confidence normal tiles must NOT appear in anomaly shortlist"

def test_shortlist_date_range_filter(in_memory_repo):
    """
    Date-range filtering restricts shortlist results to the specified ISO 8601 window.
    All sample_record fixtures default to created_at="2026-09-26T00:00:00Z".
    """
    in_memory_repo.save(sample_record("pd1", "hd1", "Highway", 0.45, priority=0.75, priority_level="HIGH", status="UNCERTAIN"))
    in_memory_repo.save(sample_record("pd2", "hd2", "River", 0.50, priority=0.70, priority_level="HIGH", status="UNCERTAIN"))

    # Window that includes the fixture date — both records should be returned
    results, total, _ = in_memory_repo.shortlist(
        mode="needs_review",
        date_from="2026-09-25T00:00:00Z",
        date_to="2026-09-27T00:00:00Z"
    )
    assert total == 2
    ids = {r.prediction_id for r in results}
    assert "pd1" in ids and "pd2" in ids

    # Future window — no records should match
    empty_results, empty_total, _ = in_memory_repo.shortlist(
        mode="needs_review",
        date_from="2026-10-01T00:00:00Z",
        date_to="2026-10-31T00:00:00Z"
    )
    assert empty_total == 0

def test_human_review_persistence(in_memory_repo):
    rec = sample_record("p-rev-1", "h-rev-1", "Highway", 0.55, status="UNCERTAIN")
    in_memory_repo.save(rec)

    review = ReviewRecord(
        prediction_id="p-rev-1",
        decision="CORRECT_CLASS",
        reviewed_class="River",
        comment="Visual river path crossing image tile.",
        reviewer_id="lead_analyst",
        reviewed_at="2026-09-27T00:00:00Z"
    )
    saved_rev = in_memory_repo.save_review(review)
    assert saved_rev.id is not None

    reviews = in_memory_repo.get_reviews_for_prediction("p-rev-1")
    assert len(reviews) == 1
    assert reviews[0].decision == "CORRECT_CLASS"
    assert reviews[0].reviewed_class == "River"
    assert reviews[0].reviewer_id == "lead_analyst"

def test_summary_statistics(in_memory_repo):
    in_memory_repo.save(sample_record("p1", "h1", "Forest", 0.80, priority=0.25, priority_level="LOW", status="ACCEPTED"))
    in_memory_repo.save(sample_record("p2", "h2", "River", 0.60, priority=0.55, priority_level="HIGH", status="UNCERTAIN"))

    summary = in_memory_repo.get_summary_statistics()
    assert summary["total_predictions"] == 2
    assert summary["average_confidence"] == 0.70
    assert summary["average_review_priority"] == 0.40
    assert len(summary["status_distribution"]) == 2
    assert len(summary["priority_distribution"]) == 2

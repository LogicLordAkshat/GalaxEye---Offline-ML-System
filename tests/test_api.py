import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.db.database import db
from app.ml.model import model_manager

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_environment():
    db.init_db()
    with db.session() as conn:
        conn.execute("DELETE FROM prediction_reviews;")
        conn.execute("DELETE FROM predictions;")
    if not model_manager.is_loaded:
        model_manager.load_model()

def create_test_image_bytes(color=(50, 100, 150)) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_status"] == "connected"
    assert data["model_loaded"] is True
    assert data["model_name"] == "landuse_cnn"
    assert data["offline_mode"] is True
    assert len(data["model_checksum"]) == 64

def test_predict_single_tile_success_and_triage_fields():
    img_bytes = create_test_image_bytes(color=(30, 120, 30))
    response = client.post(
        "/predict",
        files={"file": ("test_forest.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "tile_hash" in data
    assert "predicted_class" in data
    assert data["predicted_class"] in model_manager.classes
    assert 0.0 <= data["confidence"] <= 1.0
    
    # Triage and runner-up fields
    assert "top_2_class" in data
    assert "top_2_confidence" in data
    assert "confidence_margin" in data
    assert 0.0 <= data["confidence_margin"] <= 1.0
    assert "review_priority" in data
    assert 0.0 <= data["review_priority"] <= 1.0
    assert data["review_priority_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "review_reason" in data
    assert data["status"] in ("ACCEPTED", "UNCERTAIN")
    assert data["is_cached"] is False
    assert data["inference_latency_ms"] >= 0.0
    assert data["model_version"] == "1.0.0"
    assert len(data["probabilities"]) == 7

def test_predict_duplicate_caching():
    img_bytes = create_test_image_bytes(color=(120, 30, 30))
    
    # First upload
    res1 = client.post("/predict", files={"file": ("tile_dup.png", img_bytes, "image/png")})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["is_cached"] is False

    # Second upload (same bytes)
    res2 = client.post("/predict", files={"file": ("tile_dup.png", img_bytes, "image/png")})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["is_cached"] is True
    assert data2["tile_hash"] == data1["tile_hash"]
    assert data2["predicted_class"] == data1["predicted_class"]
    assert data2["review_priority"] == data1["review_priority"]

def test_predict_force_recompute():
    img_bytes = create_test_image_bytes(color=(200, 200, 50))
    
    # First upload
    res1 = client.post("/predict", files={"file": ("tile_force.png", img_bytes, "image/png")})
    assert res1.status_code == 200
    
    # Second upload with force_recompute=True
    res2 = client.post("/predict?force_recompute=true", files={"file": ("tile_force.png", img_bytes, "image/png")})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["is_cached"] is False

def test_predict_corrupt_file_error():
    corrupt_bytes = b"NOT_AN_IMAGE_DATA_BYTES"
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", corrupt_bytes, "image/png")}
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"]["stage"] == "decode"

def test_predict_empty_file_error():
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["stage"] == "validation"

def test_query_predictions_sorting_and_filters():
    img1 = create_test_image_bytes(color=(10, 80, 20))
    img2 = create_test_image_bytes(color=(90, 10, 80))
    
    client.post("/predict", files={"file": ("q1.png", img1, "image/png")})
    client.post("/predict", files={"file": ("q2.png", img2, "image/png")})

    # Sort by review priority
    res_sorted = client.get("/predictions?sort_by=review_priority&sort_order=desc")
    assert res_sorted.status_code == 200
    items = res_sorted.json()["items"]
    assert len(items) >= 2
    assert items[0]["review_priority"] >= items[1]["review_priority"]

def test_get_prediction_by_id_and_hash():
    img_bytes = create_test_image_bytes(color=(15, 25, 35))
    res = client.post("/predict", files={"file": ("single_fetch.png", img_bytes, "image/png")})
    data = res.json()
    pred_id = data["prediction_id"]
    tile_hash = data["tile_hash"]

    # Fetch by UUID
    res_id = client.get(f"/predictions/{pred_id}")
    assert res_id.status_code == 200
    assert res_id.json()["prediction_id"] == pred_id

    # Fetch by SHA-256 hash
    res_hash = client.get(f"/predictions/{tile_hash}")
    assert res_hash.status_code == 200
    assert res_hash.json()["tile_hash"] == tile_hash

def test_shortlist_modes():
    img_bytes = create_test_image_bytes(color=(70, 90, 110))
    client.post("/predict", files={"file": ("shortlist_tile.png", img_bytes, "image/png")})

    res = client.get("/predictions/shortlist?mode=needs_review")
    assert res.status_code == 200
    data = res.json()
    assert "criteria" in data
    assert "items" in data
    assert isinstance(data["items"], list)

def test_human_review_workflow():
    img_bytes = create_test_image_bytes(color=(40, 60, 80))
    res_pred = client.post("/predict", files={"file": ("review_tile.png", img_bytes, "image/png")})
    pred_id = res_pred.json()["prediction_id"]

    # Submit analyst review
    review_payload = {
        "decision": "CORRECT_CLASS",
        "reviewed_class": "Industrial",
        "comment": "Verified by optical aerial inspection",
        "reviewer_id": "senior_analyst"
    }
    res_rev = client.post(f"/predictions/{pred_id}/review", json=review_payload)
    assert res_rev.status_code == 201
    rev_data = res_rev.json()
    assert rev_data["prediction_id"] == pred_id
    assert rev_data["decision"] == "CORRECT_CLASS"
    assert rev_data["reviewed_class"] == "Industrial"

    # Fetch reviews log for this prediction
    res_get_revs = client.get(f"/predictions/{pred_id}/reviews")
    assert res_get_revs.status_code == 200
    assert len(res_get_revs.json()) == 1

def test_export_csv():
    img_bytes = create_test_image_bytes(color=(45, 65, 85))
    client.post("/predict", files={"file": ("export_tile.png", img_bytes, "image/png")})

    res = client.get("/predictions/export/csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "prediction_id,tile_hash,filename" in res.text
    assert "review_priority" in res.text

def test_get_predictions_summary():
    img_bytes = create_test_image_bytes(color=(33, 44, 55))
    client.post("/predict", files={"file": ("summary_tile.png", img_bytes, "image/png")})

    response = client.get("/predictions/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "average_confidence" in data
    assert "average_review_priority" in data
    assert "priority_distribution" in data
    assert "class_distribution" in data
    assert "status_distribution" in data
    assert data["model_version"] == "1.0.0"

def test_predict_batch():
    img1 = create_test_image_bytes(color=(10, 20, 30))
    img2 = create_test_image_bytes(color=(40, 50, 60))
    
    files = [
        ("files", ("batch_1.png", img1, "image/png")),
        ("files", ("batch_2.png", img2, "image/png"))
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

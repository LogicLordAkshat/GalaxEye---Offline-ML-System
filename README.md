# GalaxEye — Offline Satellite Tile Classification Service

An offline-first satellite image tile ingestion, classification, and querying service engineered for air-gapped hardware environments. The design targets efficient single-node operation; repository benchmarks are environment-specific and do not establish a universal throughput guarantee.

Built for the **GalaxEye Backend Engineer, ML Systems Take-Home Assignment**.

---
## Submission Deliverables Index

| Take-Home Requirement | File / Directory | Description |
| :--- | :--- | :--- |
| **Part 1: Design Note** | [`DESIGN.md`](./DESIGN.md) | 1–2 page comprehensive technical design note (architecture, tile lifecycle, storage, trade-offs, confidence policy, questions). |
| **Part 2: Working Slice** | [`app/`](./app)<br>[`README.md`](./README.md) | Complete working implementation with `/predict`, SQLite WAL storage, deduplication, human review, anomaly triage, and web console. |
| **Part 3: Problem Solving** | [`ANSWERS.md`](./ANSWERS.md) | Clear, grounded answers to all 4 scenario reasoning questions. |

---

## 1. System Overview

This service provides an end-to-end classification pipeline for satellite imagery:
- **100% Offline Execution:** Operates on air-gapped, isolated hardware with zero cloud or external API dependencies.
- **Local CPU Inference:** The `landuse_cnn_v1.pt` artifact is loaded by the `EuroSatCNN` architecture (model version `1.0.0`), achieving **89.05% evaluation accuracy** and **1.34 ms/tile** CPU latency.
- **Rigorous Provenance Tracking:** Every stored prediction records exact model version, SHA-256 weight checksum, preprocessing version, and per-class probability distributions.
- **Confidence & Uncertainty Policy:** Applies the configured operational threshold of 0.70, a design choice rather than a statistically calibrated probability threshold, tagging ambiguous predictions as `UNCERTAIN` for analyst triage.
- **Deduplication & Idempotency:** Employs SHA-256 tile hashing to prevent redundant inference on duplicate uploads.
- **Embedded Persistence:** Uses SQLite with Write-Ahead Logging (WAL) mode for low-overhead, concurrent, maintenance-free persistence.

---
## 2. Screenshots

### 1. Ingestion & Overview Dashboard
![Dashboard Overview]<img width="1396" height="832" alt="Screenshot 2026-09-27 192142" src="https://github.com/user-attachments/assets/9cef00a4-dae1-42ae-bf4d-5a9458c2aa5a" />

*Real-time metrics, system health, and satellite tile ingestion.*

### 2. Live Prediction & Confidence Policy
![Tile Prediction]<img width="1376" height="847" alt="Screenshot 2026-09-27 192203" src="https://github.com/user-attachments/assets/3db4cce3-f14d-4cef-b479-3223b6ca258c" />

*Tile classification, confidence score, policy threshold evaluation, and runner-up margin.*

### 3. Analyst Triage Queue & Anomaly Shortlist with Export CSV
![Export CSV]<img width="1292" height="651" alt="Screenshot 2026-09-27 192308" src="https://github.com/user-attachments/assets/e2f15c73-f685-4096-829e-553823433cf1" />

*Triage queue with one‑click filters for ACCEPTED, UNCERTAIN, and statistical anomaly detection also can get csv excel instantly.*
---

## 2. Architecture & Pipeline

```
[ Satellite Tile Upload (PNG/JPG/TIFF) ]
                  │
                  ▼
         [ FastAPI Endpoint ]
                  │
                  ▼
       [ Validation & Hashing ] ── SHA-256 Byte Hash
                  │
                  ▼
         [ Idempotency Check ] ── (If duplicate, return cached record)
                  │
                  ▼
      [ Preprocessing Pipeline ] ── Normalization [0, 1], RGB conversion, 64x64 tensor
                  │
                  ▼
     [ Local PyTorch Model (CPU) ] ── EuroSatCNN forward pass + Softmax
                  │
                  ▼
      [ Prediction Policy Engine ] ── Confidence Threshold (tau = 0.70) -> ACCEPTED / UNCERTAIN
                  │
                  ▼
      [ Prediction Repository ] ── SQLite WAL Mode, indexed queries & aggregation
                  │
                  ▼
     [ Response & Local JSON Logs ] ── Structured audit logging
```

---

## 3. Project Structure

```
d:/Galaxeye/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app lifespan, middleware, exception handlers
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # API endpoints (/predict, /predictions, /health, etc.)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic Settings & environment variables
│   │   └── logging.py           # Structured JSON log formatter
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLite connection manager with WAL mode
│   │   └── models.py            # Database record schema
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py             # PyTorch CNN architecture & ModelManager
│   │   └── inference.py         # Offline CPU inference engine & Softmax
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── prediction_repository.py # Data access layer & SQL queries
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py        # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prediction_service.py # Core orchestration service
│   │   └── preprocessing_service.py # Image validation, hashing, normalization
│   └── templates/
│       └── index.html           # Standalone Analyst Triage Console
├── candidate_tiles/             # 1,050 labeled EuroSAT candidate satellite tiles (7 classes)
├── eval_set/                    # 210 evaluation tiles
├── eval_labels.csv              # Ground truth labels for evaluation tiles
├── models/
│   ├── landuse_cnn_v1.pt        # Trained PyTorch model weights artifact
│   └── model_metadata.json      # Version, checksum, architecture metadata
├── scripts/
│   └── train_eval.py            # Offline training & evaluation script
├── tests/
│   ├── test_api.py              # API integration tests
│   ├── test_ml.py               # Model forward pass & latency tests
│   ├── test_preprocessing.py   # Image validation & corrupt file tests
│   └── test_repository.py       # Persistence, query filters, summary stats
├── DESIGN.md                    # Part 1: Official Design Note
├── ANSWERS.md                   # Part 3: Problem Solving Answers
├── requirements.txt             # Minimal pinned dependencies
├── README.md                    # Operational documentation
└── README.txt                   # EuroSAT dataset attribution
```

---

## 4. Setup & Offline Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- No GPU required (runs 100% on CPU)
- No active internet connection needed at runtime

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Retrain / Verify Offline Model
The pre-trained model artifact (`models/landuse_cnn_v1.pt`) and its metadata (`models/model_metadata.json`) are already packaged in the repository. To re-train or benchmark from candidate tiles:
```bash
python scripts/train_eval.py
```

### Step 3: Run the Service
Start the service with `uvicorn`:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`, and the analyst web console is at `http://localhost:8000`.

---

## 5. API Usage & Example Requests

### 1. Health Check
Verify system status, database connectivity, and model weights integrity:
```bash
curl -s http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "service_name": "GalaxEye Offline Satellite Tile Classification Service",
  "version": "1.0.0",
  "database_status": "connected",
  "model_loaded": true,
  "model_checksum": "ed26b8f35e251b91e7d2e1bb6732a67742eee1df5afc46fc63c7f7c196f78979",
  "offline_mode": true
}
```

---

### 2. Classify a Single Satellite Tile
Upload an image tile for classification:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@candidate_tiles/Forest/Forest_01.png"
```
**Illustrative response:**
```json
{
  "prediction_id": "3e6af3de-4d71-4b93-8461-1c9af28871ab",
  "tile_hash": "382f6ed159a8abc4109b498e18ad42d3bfe667cc679defafc1b351c78de9d2a4",
  "filename": "Forest_01.png",
  "predicted_class": "Forest",
  "confidence": 0.9517,
  "status": "ACCEPTED",
  "probabilities": {
    "AnnualCrop": 0.0054,
    "Forest": 0.9517,
    "Highway": 0.0113,
    "Industrial": 0.0001,
    "Residential": 0.0011,
    "River": 0.0278,
    "SeaLake": 0.0027
  },
  "model_name": "landuse_cnn",
  "model_version": "1.0.0",
  "model_checksum": "ed26b8f35e251b91e7d2e1bb6732a67742eee1df5afc46fc63c7f7c196f78979",
  "preprocessing_version": "1.0.0",
  "inference_latency_ms": 1.34,
  "total_latency_ms": 5.2,
  "created_at": "2026-09-27T12:00:00.000000+00:00",
  "is_cached": false
}
```

---

### 3. Duplicate Detection & Caching Demo
Sending the exact same tile again returns the cached prediction with `is_cached: true` without redundant CPU computation:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@candidate_tiles/Forest/Forest_01.png"
```
To force re-computation, pass `?force_recompute=true`:
```bash
curl -X POST "http://localhost:8000/predict?force_recompute=true" \
  -F "file=@candidate_tiles/Forest/Forest_01.png"
```

---

### 4. Query Stored Predictions
Filter predictions by class, confidence status, minimum confidence score, and pagination:
```bash
# Get all accepted predictions with confidence >= 0.80
curl "http://localhost:8000/predictions?status=ACCEPTED&min_confidence=0.80&limit=10"

# Filter specifically for 'Forest' tiles
curl "http://localhost:8000/predictions?class_name=Forest&limit=20"
```

---

### 5. Fetch Prediction by UUID or SHA-256 Hash
```bash
curl "http://localhost:8000/predictions/3e6af3de-4d71-4b93-8461-1c9af28871ab"
```

---

### 6. Observability & System Summary
Retrieve aggregate class distribution, certainty rates, average confidence, and latency:
```bash
curl "http://localhost:8000/predictions/summary"
```
**Illustrative response:**
```json
{
  "total_predictions": 150,
  "status_distribution": [
    {"status": "ACCEPTED", "count": 134, "percentage": 89.33},
    {"status": "UNCERTAIN", "count": 16, "percentage": 10.67}
  ],
  "class_distribution": [
    {"class_name": "Forest", "count": 42, "percentage": 28.0},
    {"class_name": "Highway", "count": 35, "percentage": 23.33},
    {"class_name": "Industrial", "count": 28, "percentage": 18.67}
  ],
  "average_confidence": 0.872,
  "average_inference_latency_ms": 1.34,
  "model_version": "1.0.0",
  "model_checksum": "ed26b8f35e251b91e7d2e1bb6732a67742eee1df5afc46fc63c7f7c196f78979"
}
```

---

## 6. Analyst Web Console & Screenshots

The service provides a built-in offline Web Console accessible directly at `http://localhost:8000`:

### 1. Ingestion & Overview Dashboard
![Dashboard Overview](./docs/images/dashboard_overview.png)
*Real-time metrics, system health, and satellite tile ingestion.*

### 2. Live Prediction & Confidence Policy
![Tile Prediction](./docs/images/tile_prediction.png)
*Tile classification, confidence score, policy threshold evaluation, and runner-up margin.*

### 3. Analyst Triage Queue & Anomaly Shortlist
![Analyst Triage](./docs/images/analyst_triage.png)
*Triage queue with one-click filters for ACCEPTED, UNCERTAIN, and statistical anomaly detection.*

---

## 7. Running the Test Suite

Execute the 30 unit and integration tests covering preprocessing, ML inference, SQLite persistence, caching, and API routes:
```bash
pytest -v
```

---

## 8. Key Design Choices & Trade-offs

1. **Why SQLite over PostgreSQL?**
   - In air-gapped edge hardware, running a dedicated PostgreSQL server introduces socket management, authentication secrets, port conflicts, and process crashes. SQLite is an embedded zero-configuration engine. With WAL mode enabled, it easily handles concurrent reads and serial writes with zero operational maintenance.
2. **Why In-Process Inference over Celery/Redis?**
   - Synchronous local inference keeps the offline deployment simple and avoids additional daemons. The model forward pass executes in ~1.34 ms/tile, easily satisfying local throughput without background queue worker complexity.
3. **Why `EuroSatCNN` over Generic ImageNet Models?**
   - The version `1.0.0` artifact uses `EuroSatCNN` (102k params), tailored specifically for 64×64 satellite tiles. It trains in ~15s on CPU, achieves 89.05% evaluation accuracy, and executes inference in ~1.34 ms/tile without requiring multi-gigabyte heavy dependencies.

---

## 9. Evaluation and Benchmark Scope

The independent evaluation covers 210 images from `eval_set/` using ground-truth labels in `eval_labels.csv`.

- **Evaluation Accuracy:** **89.05% (187/210 correct)**
- **Median Single-Tile CPU Latency:** **1.34 ms/tile**
- **Batch Latency:** **0.95 ms/tile**
- **Checkpoint Validation Accuracy:** **90.00%** (selected on held-out candidate split)

---

## 10. Limitations & Future Improvements

- **Fixed Spatial Extent:** Assumes 64×64 pixel tiles (~640m coverage at Sentinel-2 10m GSD). In high-resolution imagery (0.5m GSD), a 64×64 patch covers only a single building. Future work should implement dynamic GSD scaling.
- **Single-Label Constraint:** Real satellite tiles frequently contain mixed land-use (e.g., Highway crossing Forest). The next iteration should use semantic segmentation (e.g., U-Net) to output pixel-wise land cover masks.


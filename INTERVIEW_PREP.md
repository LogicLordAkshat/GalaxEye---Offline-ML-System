# GalaxEye Interview Preparation & Technical Defense Guide

This document prepares you to explain, defend, and live-code modifications to this submission during technical interviews with GalaxEye.

---

## 1. 60-Second Elevator Pitch
> *"I built an offline-first, CPU-optimized satellite tile classification service designed for air-gapped environments. It ingests image tiles via FastAPI, calculates a deterministic SHA-256 byte hash for provenance and caching, normalizes the tile, and runs the packaged `LegacyEuroSatCNN` PyTorch checkpoint on CPU. Predictions pass through a confidence policy that categorizes results into `ACCEPTED` or `UNCERTAIN` based on an operational design threshold, not a statistically calibrated cutoff. Results—including full 7-class probability distributions, model version, and weight checksums—are committed to an embedded SQLite database using WAL mode. The system provides analyst query endpoints, aggregate distribution summaries for offline drift detection, and full test coverage with zero external cloud dependencies."*

---

## 2. 5-Minute Deep Dive Walkthrough
- **Phase 1: Ingestion & Validation** — Receives PNG/TIFF tiles via `POST /predict`. The payload is checked for size limits and magic bytes before decoding. A SHA-256 hash is computed immediately from raw bytes.
- **Phase 2: Deduplication** — Before burning CPU cycles on ML inference, the repository checks if this `tile_hash` already exists in SQLite. If found, it returns the cached result with `is_cached=True`.
- **Phase 3: Preprocessing** — PIL converts arbitrary color formats (e.g. RGBA, Greyscale) to RGB, resizes/validates to 64×64 pixels, normalizes pixel intensities to `[0.0, 1.0]`, and formats the batch dimension into a `(1, 3, 64, 64)` float tensor.
- **Phase 4: ML Inference & Softmax** — The model is loaded once into memory during application startup (`lifespan`). Inference runs under `torch.no_grad()`, computing logits and Softmax probabilities across the 7 EuroSAT classes.
- **Phase 5: Uncertainty Policy** — The top confidence score is compared against $\tau = 0.70$. If $\ge \tau$, status is `ACCEPTED`; if $< \tau$, status is `UNCERTAIN`.
- **Phase 6: Persistence & Provenance** — A `PredictionRecord` is saved into SQLite containing UUID, tile hash, predicted class, confidence, JSON probability distribution, model version (`1.0.0`), model SHA-256 checksum, preprocessing version, and microsecond latencies.
- **Phase 7: Analyst Querying & Observability** — Analysts filter predictions via `GET /predictions` (by class, status, min confidence) and monitor model drift / health via `GET /predictions/summary` and `GET /health`.

---

## 3. The 20 Essential Interview Defense Points

### 1. Why this model architecture?
- **Answer:** We use `EuroSatCNN`, an optimized 3-stage CNN with BatchNorm, MaxPool, AdaptiveAvgPool, and Dropout regularization. It has 102,183 parameters, fits easily in CPU L3 cache, and achieves **89.05% evaluation accuracy** on the EuroSAT 7-class evaluation set with a median single-tile CPU forward pass latency of **1.34 ms/tile** (batch latency 0.95 ms/tile). It provides an ultra-lightweight operational footprint and deterministic offline execution.

### 2. Why SQLite instead of PostgreSQL?
- **Answer:** This service is deployed on isolated, air-gapped edge hardware. Running PostgreSQL requires managing a background daemon, local TCP ports, socket files, authentication credentials, and database migrations. SQLite is embedded directly in the application process with zero network attack surface. With Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and `busy_timeout=5000`, it supports concurrent readers alongside serialized local writes with low operational overhead. Write capacity must be benchmarked on target hardware rather than assumed.

### 3. Why offline-first?
- **Answer:** Satellite ground stations, forward edge platforms, and defense facilities operate without public internet connectivity. Relying on hosted vision APIs (OpenAI, AWS Rekognition) would introduce network failure points, bandwidth bottlenecks, and security violations.

### 4. Why confidence thresholding ($\tau = 0.70$)?
- **Answer:** Softmax outputs are prone to overconfidence on boundary or ambiguous tiles. Rather than presenting low-confidence predictions as facts, our policy tags predictions below 0.70 as `UNCERTAIN`. This allows downstream analyst systems to auto-approve high-confidence tiles and route ambiguous tiles to human review.

### 5. How does model versioning & provenance work?
- **Answer:** Every database record stores `model_name`, `model_version`, `model_checksum` (SHA-256 of the `.pt` file), and `preprocessing_version`. If the model is retrained or updated next year, historical database records remain unequivocally traceable to the exact weights file that generated them.

### 6. How are duplicate tiles handled?
- **Answer:** Preprocessing computes a SHA-256 hash of the exact input byte stream (`tile_hash`). Before invoking the ML model, the repository performs an indexed lookup on `tile_hash`. If found, the existing prediction is returned immediately (`is_cached=true`), saving CPU cycles while preserving idempotency. If an analyst wants to force re-evaluation, they pass `?force_recompute=true`.

### 7. How would you detect model degradation after 1 month offline?
- **Answer:** Through three passive signals:
  1. **Class Distribution Drift:** In SQLite, monitor class frequency over time (`GET /predictions/summary`). A sudden spike in one class indicates sensor gain shift, lens obstruction, or cloud cover.
  2. **Uncertainty Rate Drift:** A rising percentage of `UNCERTAIN` predictions indicates seasonal/atmospheric covariate shift (e.g. snow, drought).
  3. **Canary Golden Set (proposed, not implemented):** Add a local test suite that passes fixed pre-labeled tiles through the pipeline on startup to verify output consistency against golden hashes.

### 8. How would you debug wrong results in production?
- **Answer:** Execute an ordered 9-step isolation:
  1. Fetch record by UUID / tile hash (`GET /predictions/{id}`).
  2. Verify input image hash matches on-disk file.
  3. Inspect image visually for occlusion/corruption.
  4. Isolate preprocessing transformations in REPL (check tensor shape, RGB channels, `[0, 1]` scaling).
  5. Run forward pass manually to check raw logits and probability vector.
  6. Verify model weights checksum against `model_metadata.json`.
  7. Check `class_to_idx` alphabetization to rule out label swapping.
  8. Inspect SQL insertion parameters.

### 9. What is the biggest weakness of this design?
- **Answer:** Spatial resolution and scale rigidity. The model assumes 64×64 pixels representing ~10m Ground Sampling Distance (GSD). On high-resolution imagery (0.5m GSD), a 64×64 tile contains only a single car or roof, breaking the spatial context. Furthermore, satellite imagery often has mixed land use, where tile-level multi-class classification is less informative than pixel-level semantic segmentation (U-Net).

### 10. What would you improve with more time?
- **Answer:**
  1. **Multi-Spectral Ingestion:** Support 13-band Sentinel-2 GeoTIFFs (NIR, SWIR, RedEdge) with on-the-fly NDVI / NDWI computation.
  2. **Semantic Segmentation:** Transition from single-label tile classification to U-Net / SegFormer pixel segmentation.
  3. **Temperature Scaling Calibration:** Formally calibrate Softmax logits against an ECE (Expected Calibration Error) objective.
  4. **Sliding-Window Scene Ingestion:** Add an endpoint that accepts a full 10,000×10,000 satellite scene, breaks it into overlapping tiles, and stitches the resulting prediction heatmap.

### 11. What happens if throughput increases 100x?
- **Answer:**
  - *Within single node:* Convert CPU inference to **ONNX Runtime with OpenVINO / TensorRT execution providers** and batch tiles in micro-batches (e.g., batch size 32 or 64). Benchmark any resulting throughput on the target hardware before making an operational capacity claim.
  - *Multi-process:* Run Uvicorn with multiple worker processes (`uvicorn --workers 4`) behind an internal NGINX load balancer. SQLite WAL mode supports multi-process concurrent reads.

### 12. What changes if multiple edge machines need to share results?
- **Answer:** Replace SQLite with a lightweight networked database (PostgreSQL) or configure SQLite replication using tools like **LiteFS** or **rqlite** (distributed SQLite via Raft consensus) that work cleanly without cloud infrastructure.

### 13. What changes if a GPU becomes available?
- **Answer:**
  - Update `model_manager.py` to check `torch.cuda.is_available()` and set device to `cuda`.
  - Move preprocessed tensors to GPU: `tensor.to(device)`.
  - Switch from single-tile synchronous requests to a dynamic micro-batching worker queue to keep GPU CUDA cores fully saturated.

### 14. What changes if the model needs retraining?
- **Answer:** The training pipeline in `scripts/train_eval.py` is invoked with new candidate tiles. It saves a new artifact `models/landuse_cnn_v2.pt`, recalculates SHA-256 checksum, updates `model_metadata.json` (`model_version: "2.0.0"`), and restarts the service. Existing historical records retain `model_version: "1.0.0"`.

### 15. What happens if the model process crashes during inference?
- **Answer:** FastAPI global exception handlers catch the error, log the exception with traceback in structured JSON, and return HTTP 500 with `{"stage": "inference_or_persistence"}` without crashing the Uvicorn server worker.

### 16. What happens if the SQLite database becomes corrupted?
- **Answer:** SQLite WAL mode provides atomic ACID transactions that protect against power cuts. In addition, an automated cron can run `PRAGMA integrity_check;` and perform periodic `VACUUM INTO 'backup.db'` snapshots to local storage.

### 17. What happens if input image distribution changes (e.g. winter snow)?
- **Answer:** The model outputs will show a surge in `UNCERTAIN` status and probability entropy. The observability summary endpoint (`GET /predictions/summary`) reveals this drop in average confidence, signaling to analysts that seasonal fine-tuning or domain adaptation is needed.

### 18. What if an analyst needs to query by confidence range and date?
- **Answer:** The repository includes SQL indexes on `status`, `predicted_class`, `model_version`, and `created_at`. Extending the query method to add `start_date` and `end_date` parameters is a 3-line modification in `app/repositories/prediction_repository.py`.

### 19. Why not return an HTTP 400 error when confidence is low?
- **Answer:** Because low confidence is an ML property of the imagery, not an HTTP or client transport failure. Returning HTTP 200 with `status: "UNCERTAIN"` and full probabilities allows downstream consumer services to ingest, store, and route the record for triage rather than crashing client retry loops.

### 20. How is code modularity enforced?
- **Answer:** We enforce strict separation of concerns across 5 isolated layers:
  `API (Routes)` -> `Service (Domain Orchestration)` -> `ML (Inference Engine)` -> `Repository (SQL Data Access)` -> `Database (Connection Manager)`.

---

## 4. Live Coding Modifications an Interviewer Might Ask You to Make

### Scenario A: "Add a filter for date range to `GET /predictions`"
1. In `app/schemas/prediction.py`, add `start_date: Optional[str] = None` and `end_date: Optional[str] = None` to `PredictionQueryFilter`.
2. In `app/repositories/prediction_repository.py`:
   ```python
   if start_date:
       conditions.append("created_at >= ?")
       params.append(start_date)
   if end_date:
       conditions.append("created_at <= ?")
       params.append(end_date)
   ```
3. In `app/api/routes.py`, add `start_date` and `end_date` query parameters to `get_predictions`.

### Scenario B: "Make the confidence threshold configurable at runtime per request"
1. In `app/api/routes.py`, add `threshold: Optional[float] = Query(None, ge=0.0, le=1.0)` to `predict_tile`.
2. Pass `threshold` to `prediction_service.predict_tile(..., threshold=threshold)`.
3. In `prediction_service.py`:
   ```python
   applied_threshold = threshold if threshold is not None else settings.CONFIDENCE_THRESHOLD
   status = "ACCEPTED" if confidence >= applied_threshold else "UNCERTAIN"
   ```

### Scenario C: "Add an endpoint to export predictions as CSV for an analyst"
1. In `app/api/routes.py`, add `GET /predictions/export/csv`.
2. Fetch records via `prediction_repository.query(limit=10000)`.
3. Use Python's built-in `csv.writer` with `StreamingResponse(iter([csv_string]), media_type="text/csv")`.

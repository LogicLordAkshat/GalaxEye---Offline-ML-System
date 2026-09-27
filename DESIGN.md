# GalaxEye Backend / ML Systems Take-Home Design Note

## 1. Problem Understanding

GalaxEye requires an **offline satellite tile classification service** operating in air-gapped, isolated hardware environments without external network or cloud connectivity. The system must ingest satellite image tiles, perform local computer vision classification across land-use classes, persist prediction records with rich provenance, and provide querying capabilities for downstream analysts.

Key operational realities:
- **Offline isolation:** Zero hosted cloud vision or LLM APIs (OpenAI, Claude, GCP/AWS). All weights and dependencies reside locally.
- **CPU execution:** Fast, deterministic local inference without hard GPU dependencies.
- **Provenance & Auditability:** Every classification must track exact model weights, preprocessing pipelines, and confidence metrics to support historical reproducibility.
- **Uncertainty awareness:** The service must acknowledge ML model fallibility, flagging ambiguous predictions rather than silently asserting false certainty.

---

## 2. Proposed Architecture

The system is designed as a modular, layered service separating network boundaries, domain business logic, machine learning inference, and database persistence.

```
                           +------------------------+
                           |  Analyst / Ingestion   |
                           +------------------------+
                                    |        ^
                   Tile Upload (POST)        | Query / Summary (GET)
                                    v        |
                           +------------------------+
                           |    FastAPI Router      |
                           +------------------------+
                                       |
                                       v
                           +------------------------+
                           |  Prediction Service    |
                           +------------------------+
                            /          |           \
                           v           v            v
            +-----------------+  +------------+  +----------------------+
            | Preprocessing   |  | ML Engine  |  | Policy Engine        |
            | (Validate/Hash) |  | (PyTorch)  |  | (Confidence/Status)  |
            +-----------------+  +------------+  +----------------------+
                                       |
                                       v
                           +------------------------+
                           | Prediction Repository  |
                           +------------------------+
                                       |
                                       v
                           +------------------------+
                           |  SQLite (WAL Enabled)  |
                           +------------------------+
```

### Tile Lifecycle Walkthrough
1. **Upload & Ingestion:** The client transmits a tile (e.g. PNG/TIFF) via `POST /predict`.
2. **Validation & Hashing:** Image headers and magic bytes are verified. A deterministic SHA-256 hash of the exact input byte stream is generated.
3. **Idempotency / Cache Lookup:** The repository checks if this `tile_hash` was already processed under the current model version. If present, the cached result is returned immediately, saving CPU cycles.
4. **Preprocessing:** Image is converted to RGB 3-channel format, verified/resized to 64×64 pixels, normalized to `[0.0, 1.0]`, and shaped into a `(1, 3, 64, 64)` float tensor.
5. **Local Inference:** PyTorch forward pass runs on CPU under `torch.no_grad()`, computing class logits and Softmax probabilities.
6. **Confidence Policy:** The top probability is evaluated against a configured threshold ($\tau = 0.70$). Predictions meeting $\ge \tau$ are tagged `ACCEPTED`; others are tagged `UNCERTAIN`.
7. **Persistence:** A `PredictionRecord` containing UUID, tile hash, predicted class, confidence, full probability distribution, model version, model SHA-256 checksum, preprocessing version, and latencies is committed to SQLite.
8. **Response:** Structured JSON response is returned to the client. API latency depends on request handling, image decoding, persistence, and the host environment; no universal end-to-end latency guarantee is claimed.

---

## 3. Component Responsibilities

| Component | Responsibility | Boundary Separation |
| :--- | :--- | :--- |
| **API Layer (`app/api/`)** | HTTP routing, request parsing, status codes, OpenAPI schema serialization. | No direct ML tensor manipulation or SQL queries. |
| **Preprocessing (`app/services/`)** | Payload validation, byte hashing, PIL decode, color mode normalization, tensor transformations. | Pure deterministic transformation; zero database access. |
| **ML Engine (`app/ml/`)** | Model architecture definition, weights loading on CPU, forward pass execution, Softmax distribution. | Decoupled from HTTP and database schemas. |
| **Prediction Policy (`app/services/`)** | Decision logic for prediction acceptance, threshold evaluation, and uncertainty tagging. | Encapsulated domain business rule. |
| **Repository (`app/repositories/`)** | Data access layer, SQL query generation, transactions, aggregation queries. | Hides SQLite specifics from service layer. |
| **Database (`app/db/`)** | Connection lifecycle, SQLite WAL mode, schema initialization, table indexing. | Infrastructure layer. |
| **Observability (`app/core/`)** | Structured JSON logging with event tags, latencies, and operational health checks. | Cross-cutting utility. |

---

## 4. Storage Design

The system uses an embedded SQLite database configured with **Write-Ahead Logging (WAL)**.

### Schema: `predictions` Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,      -- UUIDv4 identifier
    tile_hash TEXT NOT NULL,                 -- SHA-256 hash of raw tile bytes (Index)
    filename TEXT NOT NULL,                  -- Original client filename
    predicted_class TEXT NOT NULL,           -- Top class name (Index)
    confidence REAL NOT NULL,                -- Softmax score [0.0 - 1.0]
    status TEXT NOT NULL,                    -- 'ACCEPTED' | 'UNCERTAIN' (Index)
    all_probabilities TEXT NOT NULL,         -- Full class distribution as JSON string
    model_name TEXT NOT NULL,                -- 'landuse_cnn'
    model_version TEXT NOT NULL,             -- Semantic version e.g. '1.0.0' (Index)
    model_checksum TEXT NOT NULL,            -- SHA-256 hash of model weights file
    preprocessing_version TEXT NOT NULL,     -- Preprocessing pipeline version
    inference_latency_ms REAL NOT NULL,      -- Pure model forward pass time
    total_latency_ms REAL NOT NULL,          -- End-to-end request latency
    created_at TEXT NOT NULL                 -- ISO 8601 UTC timestamp (Index)
);
```

### Why These Fields Exist:
- **`tile_hash`**: Guarantees content-based deduplication and image data integrity.
- **`model_version` + `model_checksum`**: Ensures absolute traceability. If a model is updated or replaced, historical rows remain explicitly tied to the exact weights file that created them.
- **`all_probabilities`**: Essential for secondary triage. An analyst can inspect whether a misclassified "Highway" had a 40% secondary probability for "River".
- **`status`**: Enables analysts to query only verified/high-confidence tiles or extract uncertain tiles for manual review.

---

## 5. Confidence and Uncertainty Handling

Raw Softmax scores output by neural networks often exhibit overconfidence on out-of-distribution inputs. Our prediction policy handles this deliberately:

1. **Threshold Partitioning:** A default operational threshold ($\tau = 0.70$) partitions predictions:
   $$\text{Status} = \begin{cases} \text{ACCEPTED}, & \text{if } \max(P) \ge 0.70 \\ \text{UNCERTAIN}, & \text{if } \max(P) < 0.70 \end{cases}$$
2. **Distribution Persistence:** Rather than discarding secondary scores, the full 7-class probability vector is persisted as structured JSON.
3. **Calibration Limitations:** The 0.70 threshold is an operational design choice, not a statistically calibrated probability cutoff. Softmax output is a relative score, not a true Bayesian posterior probability, unless calibrated via Temperature Scaling or Isotonic Regression against a holdout dataset.

---

## 6. Architectural Trade-offs

### A. Embedded SQLite vs. Client-Server PostgreSQL
- *Decision:* **SQLite with WAL mode**.
- *Rationale:* For an offline, single-node edge deployment with zero external network access, SQLite eliminates daemon management, external ports, socket failures, and authentication overhead. WAL mode allows concurrent readers alongside active writes with zero operational maintenance.
- *Trade-off:* Limited concurrent write throughput compared to a multi-node database. However, satellite tile classification on a single edge machine is compute-bound (CPU inference), not disk write-bound.

### B. Synchronous In-Process Inference vs. Asynchronous Queue (Celery/Redis)
- *Decision:* **Synchronous In-Process Execution**.
- *Rationale:* Synchronous local inference keeps the offline deployment simple and avoids additional daemons. In one verification environment, the packaged model-only CPU forward pass measured mean 5.38 ms/tile, p50 3.99 ms/tile, and p95 14.23 ms/tile. These measurements exclude end-to-end API latency and are not universal guarantees.
- *Trade-off:* If bulk multi-gigabyte imagery bursts occur, clients must use batch uploads or parallel HTTP workers.

### C. Lightweight Dedicated CNN vs. Heavy Pretrained Foundation Models
- *Decision:* **`EuroSatCNN` (3-stage CNN with BatchNorm & Dropout, 102k params), model version `1.0.0`**.
- *Rationale:* The trained artifact is a compact CNN tailored for 64×64 satellite tiles running offline on CPU. It achieves **89.05% evaluation accuracy** on the EuroSAT 7-class evaluation set with a median single-tile CPU forward pass latency of **1.34 ms/tile** (batch latency 0.95 ms/tile).
- *Trade-off:* Lower representation capacity than a multi-gigabyte foundation model, but provides an ultra-lightweight operational footprint, zero GPU requirements, and fast deterministic CPU execution on edge nodes.

### D. Prediction Policy: Status Flagging vs. Hard Rejection
- *Decision:* **Persist with `UNCERTAIN` status rather than rejecting with HTTP 4xx**.
- *Rationale:* An ambiguous satellite tile is not a client request error. Persisting uncertain predictions allows analysts to query and audit ambiguous regions and discover edge cases or new terrain types.

### E. Image Storage: Store Metadata + Hash vs. Raw Binary Blobs
- *Decision:* **Store SHA-256 hash and metadata in SQLite; do not bloat SQLite with raw image BLOBs**.
- *Rationale:* Storing raw images in relational tables rapidly inflates database size, slowing index lookups and backup operations. The deterministic hash links the database record to the tile file on disk.

---

## 7. Evaluation and Benchmark Scope

The independent evaluation uses 210 images from `eval_set/` with ground-truth labels from `eval_labels.csv`.

- **Evaluation Accuracy:** **89.05% (187/210 correct)**
- **Median CPU Forward Latency:** **1.34 ms/tile** (batch latency 0.95 ms/tile)
- **Validation Checkpoint Accuracy:** **90.00%** (selected on held-out candidate split)

## 8. Offline Constraints

The service satisfies complete offline isolation:
1. **Zero External Calls:** No cloud APIs, telemetry pings, remote model downloads, or external network dependencies.
2. **Local Artifacts:** PyTorch model weights (`models/landuse_cnn_v1.pt`) and schema metadata (`models/model_metadata.json`) are bundled locally.
3. **Self-Contained Lifespan:** On application startup, weights are read directly from local disk into CPU memory.

---

## 8. Assumptions

1. **Input Format:** Ingested tiles are 2D optical images (PNG, JPG, TIFF) representing standard 3-channel RGB bands.
2. **Tile Size:** Tiles are centered around 64×64 pixels (EuroSAT resolution) or can be resampled to 64×64 without significant semantic loss.
3. **Target Vocabulary:** The land-use classification is closed-set across the 7 EuroSAT classes (`AnnualCrop`, `Forest`, `Highway`, `Industrial`, `Residential`, `River`, `SeaLake`).
4. **Hardware Environment:** Runs on standard x86_64 or ARM64 edge machines with at least 1 CPU core and 512MB RAM.
5. **Idempotency Policy:** Re-uploading an identical tile (matching SHA-256) returns the existing prediction unless `force_recompute=true` is specified.

---

## 9. Key Questions for GalaxEye

1. **Sensor & Band Modalities:** What specific sensor bands are captured (RGB vs 13-band Sentinel-2 L1C/L2A vs SAR imagery)? Should the preprocessor support multi-spectral TIFFs with NIR/RedEdge channels (e.g. NDVI calculation)?
2. **Resolution & GSD:** What is the Ground Sampling Distance (GSD) of incoming satellite tiles (e.g., 10m/pixel vs 0.5m/pixel)? Will tiles require sliding-window tiling over large swaths?
3. **Multi-label / Mixed Pixels:** Satellite tiles often contain mixed land cover (e.g., a highway running through a forest). Should the system transition from multi-class single-label to multi-label segmentation/classification?
4. **Operational Latency & Throughput Targets:** What is the expected ingestion volume (e.g., 10 tiles/sec vs 100,000 tiles per orbital pass)?
5. **Uncertainty Workflow:** What is the operational protocol when a tile is marked `UNCERTAIN`—should it trigger automated human-in-the-loop analyst queuing?
6. **Data Retention & Storage Quotas:** How long should historical prediction records and image tiles be retained on edge hardware before archiving or pruning?

# GalaxEye Take-Home: Part 3 — Problem Solving Answers

---

### Question 1:
**"Your classifier turns out to be wrong about 30% of the time. What do you do — and how would you even decide whether it’s 'good enough' to be useful at all?"**

#### 1. Why "30% Wrong" is Insufficient Information
A flat 70% overall accuracy metric obscures critical operational trade-offs:
- **Asymmetric Cost of Errors:** In satellite analytics, confusing `Forest` with `AnnualCrop` is often a minor visual boundary error. In contrast, mistaking an `Industrial` facility for `SeaLake` or missing critical infrastructure (`Highway`) carries a high operational penalty.
- **Class Imbalance & Base Rates:** If 80% of incoming tiles are agricultural fields (`AnnualCrop`), a naive majority-class classifier achieves 80% accuracy while being completely useless for all other classes.
- **Confidence Separation:** If the model is wrong 30% of the time, but all those errors occur with low confidence ($P < 0.55$), while correct predictions have high confidence ($P > 0.85$), the model is highly actionable simply by applying a confidence threshold.

#### 2. Determining "Good Enough" for Analyst Workflow
To decide if the model is useful, I evaluate it against the specific downstream workflow:
- **Baseline Comparison:** Does 70% accuracy beat the existing alternative (e.g., random guess = 14.3% across 7 classes, or zero-automation manual scanning)?
- **Hypothetical workload illustration (not measured):** If an analyst currently reviews 10,000 tiles manually, and a future validated operating point correctly accepts 6,000 high-confidence tiles with >95% precision, the analyst would inspect the remaining 4,000 `UNCERTAIN` tiles. That hypothetical scenario would reduce manual review volume by 60%; this repository does not measure or claim a 60% labor reduction.
- **Per-Class Precision, Recall, and Confusion Matrix:** We inspect the full confusion matrix. If the model has 93% precision on `Residential` and `Industrial` (as observed on our EuroSAT eval set), those predictions can be auto-approved, routing only ambiguous classes (`River` vs `Highway`) to human review.

#### 3. Concrete Remediation Steps
1. **Calibrate Confidence Thresholds:** Use temperature scaling or validation ROC curves to tune class-specific thresholds so that accepted predictions meet a target precision (e.g., 95%).
2. **Error Pattern Diagnosis:** Inspect the confusion matrix to identify specific failure modes (e.g., narrow rivers mistaken for linear highways due to 64x64 resolution limits).
3. **Data Augmentation & Targeted Training:** Augment the underperforming classes with domain-specific transforms (random rotations, color jitter for seasonal changes, spectral index features like NDVI).
4. **Active Learning Feedback Loop:** Flag `UNCERTAIN` predictions for analyst labeling, adding them to future retraining cycles.

---

### Question 2:
**"This service runs offline, with no one watching it live. A month after deployment, how would you know it’s still working correctly?"**

#### 1. Distinction: Service Health vs. Model Quality
A system can return HTTP 200 responses with zero server errors while making completely garbage predictions due to domain shift or sensor degradation. Offline observability must monitor both dimensions.

#### 2. Automated Offline Auditing Mechanisms
Because the system cannot send alerts over the internet, we inspect local health signals stored in SQLite and structured log files:

1. **System & Operational Health:**
   - **Error Stage Counts:** Check `validation_error`, `corrupt_image_error`, and `db_error` counts in structured JSON logs.
   - **Throughput & Latency Drift:** Monitor 95th/99th percentile inference latency. Compare it with a recorded baseline from the same deployment environment; a substantial increase may indicate memory pressure or disk I/O thrashing.
   - **Storage & Disk Headroom:** Verify SQLite database file size and available disk space to prevent silent write failures.

2. **Model Distribution & Covariate Drift:**
   - **Class Output Distribution Drift:** In SQLite, query the class distribution over the last 30 days (`GET /predictions/summary`). If `SeaLake` suddenly shifts from 10% to 75% of predictions, the sensor may have camera exposure corruption, cloud cover obstruction, or defective gain settings.
   - **Confidence Score Degradation:** Track the percentage of `UNCERTAIN` predictions over time. A rising uncertainty rate (e.g. from 15% to 55%) is an early indicator of distribution shift (e.g., winter snow cover, seasonal vegetation change, or differing sun angles).

3. **Golden Validation Canary Test:**
   - Deploy an offline cron task or startup test that runs a fixed, immutable "canary suite" of 20 pre-labeled satellite tiles through the pipeline.
   - Compare the output against known golden hashes and expected classes. If accuracy on the canary set drops or outputs diverge from the expected checksum, model corruption or runtime regression has occurred.

---

### Question 3:
**"Tiles are coming in fine, but the stored results look wrong. Walk through how you would find the cause — your actual steps, in order."**

To systematically isolate where the pipeline is failing, I execute the following ordered diagnostic workflow:

```
[1. Identify Problematic Record]
              ↓
[2. Verify Tile Hash & Raw Bytes]
              ↓
[3. Inspect Visual Image Tile]
              ↓
[4. Isolate Preprocessing Transformations]
              ↓
[5. Run Isolated Model Forward Pass]
              ↓
[6. Inspect Full Probability Vector]
              ↓
[7. Verify Model Weights Checksum & Version]
              ↓
[8. Check Database Mapping & Write Logic]
              ↓
[9. Compare Against Structured Logs]
```

#### Step-by-Step Diagnostic Sequence:
1. **Retrieve the Problematic Record:** Query the exact database record using its `prediction_id` or `tile_hash` (`GET /predictions/{id}`). Inspect `model_version`, `model_checksum`, `confidence`, `status`, and `all_probabilities`.
2. **Verify Input Data Integrity:** Hash the raw input file on disk using SHA-256 and confirm it matches `tile_hash`. Ensure the file was not partially transferred or corrupted.
3. **Visually Inspect the Image:** Open the tile. Is it solid black, overexposed, occluded by 100% cloud cover, or formatted with unexpected multi-spectral channels (e.g. 16-bit TIFF misinterpreted as 8-bit PNG)?
4. **Isolate Preprocessing Transformation:** Run the `preprocessing_service` on the tile in a Python REPL. Check tensor shape `(1, 3, 64, 64)`, min/max pixel values (`0.0` to `1.0`), channel ordering (RGB vs BGR vs RGBA), and normalization divisor (`/ 255.0`).
5. **Execute Raw Model Inference:** Pass the preprocessed tensor into `model_manager.model(tensor)` directly. Check raw logit values before Softmax.
6. **Analyze Class Probability Distribution:** Review `all_probabilities`. Is the model uncertain between two similar classes (e.g., 48% `River` vs 52% `Highway`), or is it assigning 99% probability to an absurd class?
7. **Verify Model Weights Integrity:** Compute the SHA-256 checksum of `models/landuse_cnn_v1.pt` and compare it against `model_checksum` in `models/model_metadata.json`. This confirms no corrupted weights, truncated files, or silent model swaps.
8. **Inspect Class Index Mapping:** Verify that `class_to_idx` matches the alphabetical order used during training. (A common bug is class list sorting mismatch between training script and inference service, causing label swapping).
9. **Check Database Persistence Layer:** Compare the dictionary returned by `inference_engine` with the values written into the SQLite row to rule out column transposition during SQL `INSERT`.

---

### Question 4:
**"What’s the weakest part of your design, and what would break it first?"**

#### 1. The Weakest Component: Resolution Sensitivity & Fixed Spatial Scale
The weakest part of this design is the assumption of a **fixed 64×64 pixel spatial input with single-label classification**.

#### 2. What Would Break It First in Real Satellite Operations
- **Ground Sampling Distance (GSD) Mismatch:** The model was trained on EuroSAT (Sentinel-2 imagery at ~10 meters/pixel resolution, where a 64×64 tile covers 640m × 640m of terrain). If the deployment receives high-resolution commercial imagery (e.g. PlanetScope at 3m/pixel or WorldView at 0.3m/pixel), a 64×64 crop will contain only a single rooftop or tree canopy rather than a recognizable "Residential" neighborhood or "Forest". The model will fail completely because the spatial context is radically shifted.
- **Mixed Land-Use & Boundary Tiles:** Real-world satellite tiles rarely contain 100% pure homogenous land use. A single tile frequently contains a highway cutting through farmland near a river. Forcing a single discrete class label with standard multi-class Softmax causes arbitrary predictions and high uncertainty on boundary tiles.

#### 3. Next Engineering Iteration
1. **Multi-Resolution Pyramids / Patch Tiling:** Implement dynamic spatial tiling that accepts arbitrary large-scale GeoTIFF scenes and extracts tiles scaled according to their physical Ground Sampling Distance metadata.
2. **Transition to Semantic Segmentation (U-Net / DeepLab):** Move from tile-level classification to pixel-level semantic segmentation, producing pixel-wise land-cover masks rather than single categorical labels.
3. **Multi-Spectral & Sensor Ingestion:** Upgrade preprocessing to ingest raw 13-band Sentinel-2 or SAR data with automatic radiometric calibration and NoData masking.

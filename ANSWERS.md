# Part 3 — Problem Solving & Reasoning

Here are my direct answers and reasoning for the four scenario questions.

---

### 1. Your classifier turns out to be wrong about 30% of the time. What do you do — and how would you even decide whether it’s "good enough" to be useful at all?

If my model is 70% accurate overall, the first thing I would do is look past that aggregate number. In real satellite systems, aggregate accuracy doesn't tell you if the system is actually useful or dangerous.

Here is how I would evaluate and handle it:

1. **Check error cost asymmetry:** Confusing `Forest` with `AnnualCrop` along a field boundary is usually a mild visual error. But mistaking an `Industrial` depot for `SeaLake` or missing critical infrastructure like a `Highway` is a costly mistake. I'd evaluate per-class precision and recall rather than overall accuracy.
2. **Check confidence separation:** If the 30% errors happen when the model has low confidence (say $P < 0.55$), while correct predictions have high confidence ($P > 0.85$), the model is already very useful. I can set a threshold ($\tau = 0.70$) so high-confidence predictions pass through automatically, and only low-confidence ones are routed to an analyst.
3. **Deciding if it's "good enough":** I compare it against the real-world baseline. If an analyst currently has to manually scan 10,000 tiles by hand with zero automation, and this model can reliably filter out 6,000 obvious tiles with 95%+ precision, it saves huge amounts of manual labor. If it provides a net reduction in human cognitive fatigue without dropping mission-critical targets, it is useful.
4. **Action plan to improve it:**
   - Calibrate confidence scores (e.g. Temperature Scaling) so probabilities reflect real accuracy.
   - Inspect the confusion matrix to see specific failure pairs (e.g. narrow rivers vs linear highways).
   - Augment training data specifically for the confused classes (rotations, brightness variations, seasonal color shifts).
   - Use the analyst review feedback from the triage queue to collect hard negative examples for the next retraining round.

---

### 2. This service runs offline, with no one watching it live. A month after deployment, how would you know it’s still working correctly?

Because this runs completely offline with no live telemetry, I can't rely on cloud dashboards or Datadog alerts. I have to design passive, local signals that build up in the SQLite database and structured log files.

Here is how I would verify system and model health after a month:

1. **Check class distribution drift:** Using the `GET /predictions/summary` endpoint or a quick SQL query, I'd check the frequency of predicted classes over time. If `SeaLake` normally makes up 10% of tiles but suddenly jumped to 70% in week 3, that immediately tells me something went wrong with the sensor (e.g., lens occlusion, severe gain adjustment, or persistent cloud cover).
2. **Monitor the uncertainty rate:** I'd track the percentage of predictions flagged as `UNCERTAIN` over time. If the uncertainty rate climbs steadily from 15% to 60%, it indicates severe covariate shift — like seasonal changes (winter snow cover, autumn foliage) or different atmospheric conditions that the training set never saw.
3. **Inspect error logs & latency percentiles:** Check local structured JSON logs for spikes in `validation_error`, `corrupt_image_error`, or database locks. Also monitor p95 and p99 inference latencies to make sure the CPU isn't throttling or running out of memory.
4. **Run an immutable golden canary test:** I'd keep a small local set of 20 fixed, pre-labeled tiles on disk that runs through the model on service startup or via a local cron. If the output logits or class predictions on those 20 golden tiles ever diverge from the expected checksum, I know immediately that the model weights or runtime libraries got corrupted.

---

### 3. Tiles are coming in fine, but the stored results look wrong. Walk us through how you’d find the cause — your actual steps, in order.

When inputs arrive but outputs look broken, I trace the tile through the pipeline step by step to isolate exactly where corruption happened:

1. **Pull the exact record:** I'd take the `prediction_id` or `tile_hash` and query `GET /predictions/{id}` to inspect what was actually saved — the predicted class, confidence, full probability distribution, model version, and model checksum.
2. **Verify input file integrity:** Compute the SHA-256 hash of the raw tile on disk and match it against the stored `tile_hash` to make sure the file wasn't truncated or corrupted during transfer.
3. **Visually open the raw image:** Check if the image itself is pure black, overexposed, 100% white cloud cover, or formatted with weird multi-spectral bands (like a 16-bit GeoTIFF misinterpreted as an 8-bit PNG).
4. **Step through preprocessing in Python:** In a local REPL, run `preprocessing_service.validate_and_preprocess()` on that tile. Inspect the tensor shape `(1, 3, 64, 64)`, min/max values (`[0.0, 1.0]`), channel order (RGB vs BGR), and division by 255.0.
5. **Run a manual model forward pass:** Pass the preprocessed tensor directly into `model_manager.model(tensor)` and check the raw logits before Softmax.
6. **Inspect the full probability vector:** Look at all 7 class scores. Is the model split 50/50 between two similar classes (like `River` and `Highway`), or is it completely confident in a wrong class?
7. **Verify model weight checksum:** Hash the on-disk `.pt` file using SHA-256 and compare it to `model_metadata.json`. This proves nobody accidentally replaced or truncated the weights artifact.
8. **Check the label vocabulary mapping:** Verify that `class_to_idx` in `app/ml/model.py` exactly matches the alphabetical index order used during training. A mismatch here is a classic bug that swaps class names silently.
9. **Check the database insert statement:** Compare the dictionary returned by inference against what was written to SQLite to confirm columns weren't transposed during the SQL insert.

---

### 4. What’s the weakest part of your design, and what would break it first?

The weakest part of this design is the assumption of a **fixed 64×64 pixel spatial input with single-label classification**.

Here is what would break it first in a production satellite mission:

1. **Ground Sampling Distance (GSD) mismatch:** The model was trained on EuroSAT (Sentinel-2 at ~10m per pixel, where a 64×64 tile covers 640m × 640m of terrain). If the system is deployed on high-resolution commercial imagery (e.g. WorldView at 0.3m/pixel or PlanetScope at 3m/pixel), a 64×64 crop covers only a single rooftop or tree canopy instead of a recognizable "Residential" neighborhood or "Forest". The spatial context completely breaks down, and the classifier will produce nonsense.
2. **Mixed land-use on tile boundaries:** Real satellite swaths don't fit into neat discrete categories. A single tile often has a highway running through farmland with a river nearby. Forcing a single discrete class label with Softmax makes the model struggle on every mixed boundary tile.

**How I would fix this in the next iteration:**
- Implement dynamic sliding-window tiling that extracts crops scaled to the physical Ground Sampling Distance (GSD) metadata of the input scene.
- Transition from tile-level classification to pixel-level semantic segmentation (like a lightweight U-Net), which produces continuous land-cover masks rather than single categorical labels.

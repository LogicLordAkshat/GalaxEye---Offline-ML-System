import os
import json
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    print(f"Connecting to live server at {BASE_URL}...\n")

    # 1. Health Check
    banner("1. Health Check (GET /health)")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {res.status_code}")
    print(json.dumps(res.json(), indent=2))

    # 2. Predict Sample Satellite Tiles
    banner("2. Classify Satellite Image Tiles (POST /predict)")
    sample_tiles = [
        ("candidate_tiles/Forest/Forest_01.png", "Forest_01.png"),
        ("candidate_tiles/Highway/Highway_01.png", "Highway_01.png"),
        ("candidate_tiles/Industrial/Industrial_01.png", "Industrial_01.png"),
        ("candidate_tiles/SeaLake/SeaLake_01.png", "SeaLake_01.png"),
        ("eval_set/tile_001.png", "tile_001.png"),
        ("eval_set/tile_005.png", "tile_005.png")
    ]

    stored_prediction = None
    for path, name in sample_tiles:
        if os.path.exists(path):
            with open(path, "rb") as f:
                t0 = time.time()
                res = requests.post(f"{BASE_URL}/predict", files={"file": (name, f.read(), "image/png")})
                elapsed = (time.time() - t0) * 1000
                data = res.json()
                if stored_prediction is None:
                    stored_prediction = data
                print(f"[{data.get('predicted_class', 'UNKNOWN'):11s}] Conf: {data.get('confidence', 0):.4f} | Status: {data.get('status', 'N/A'):9s} | Cached: {str(data.get('is_cached')):5s} | Roundtrip: {elapsed:.1f}ms | Tile: {name}")

    # 3. Duplicate Detection Demo
    banner("3. Duplicate Tile Detection & Caching Demo (POST /predict)")
    print("Uploading 'Forest_01.png' again (identical bytes)...")
    with open("candidate_tiles/Forest/Forest_01.png", "rb") as f:
        t0 = time.time()
        res = requests.post(f"{BASE_URL}/predict", files={"file": ("Forest_01.png", f.read(), "image/png")})
        elapsed = (time.time() - t0) * 1000
        data = res.json()
        print(f"Status Code: {res.status_code}")
        print(f"Response (is_cached={data['is_cached']}):")
        print(json.dumps(data, indent=2))
        print(f"-> Returned in {elapsed:.2f}ms directly from SQLite index without model forward pass!")

    # 4. Force Recompute Demo
    banner("4. Force Recompute on Duplicate (POST /predict?force_recompute=true)")
    with open("candidate_tiles/Forest/Forest_01.png", "rb") as f:
        res = requests.post(f"{BASE_URL}/predict?force_recompute=true", files={"file": ("Forest_01.png", f.read(), "image/png")})
        data = res.json()
        print(f"Response (is_cached={data['is_cached']}, inference_latency={data['inference_latency_ms']}ms):")
        print(f"Predicted: {data['predicted_class']} ({data['confidence']}) - Status: {data['status']}")

    # 5. Query Stored Predictions with Filters
    banner("5. Analyst Querying (GET /predictions?limit=5)")
    res = requests.get(f"{BASE_URL}/predictions?limit=5")
    data = res.json()
    print(f"Total matching records in DB: {data['total']}")
    print(f"Returned in page: {data['count']}")
    for item in data['items']:
        print(f" - ID: {item['prediction_id'][:8]}... | Class: {item['predicted_class']:11s} | Conf: {item['confidence']:.4f} | Status: {item['status']:9s} | Hash: {item['tile_hash'][:12]}...")

    # Filter by specific class
    banner("5b. Filter by Class (GET /predictions?class_name=Industrial)")
    res = requests.get(f"{BASE_URL}/predictions?class_name=Industrial")
    data = res.json()
    print(f"Found {data['total']} Industrial tiles:")
    for item in data['items']:
        print(f" - {item['filename']} -> {item['predicted_class']} ({item['confidence']}) [{item['status']}]")

    # 6. Fetch Single Prediction by UUID and Hash
    if stored_prediction:
        pred_id = stored_prediction["prediction_id"]
        tile_hash = stored_prediction["tile_hash"]
        banner(f"6. Fetch Prediction by UUID (GET /predictions/{pred_id})")
        res = requests.get(f"{BASE_URL}/predictions/{pred_id}")
        print(json.dumps(res.json(), indent=2))

        banner(f"6b. Fetch Prediction by SHA-256 Hash (GET /predictions/{tile_hash})")
        res = requests.get(f"{BASE_URL}/predictions/{tile_hash}")
        print(f"Found tile hash {tile_hash[:16]}... -> Predicted: {res.json()['predicted_class']}")

    # 7. Observability Summary
    banner("7. Offline Observability Summary (GET /predictions/summary)")
    res = requests.get(f"{BASE_URL}/predictions/summary")
    print(json.dumps(res.json(), indent=2))

    # 8. Error Handling Demonstrations
    banner("8. Error Handling & Validation Tests")
    
    print("\nA. Sending empty file (0 bytes):")
    res = requests.post(f"{BASE_URL}/predict", files={"file": ("empty.png", b"", "image/png")})
    print(f"Status: {res.status_code} -> {res.json()}")

    print("\nB. Sending corrupt non-image bytes:")
    res = requests.post(f"{BASE_URL}/predict", files={"file": ("corrupt.png", b"CORRUPT_NOT_AN_IMAGE", "image/png")})
    print(f"Status: {res.status_code} -> {res.json()}")

    print("\nC. Requesting non-existent prediction ID:")
    res = requests.get(f"{BASE_URL}/predictions/non-existent-uuid-12345")
    print(f"Status: {res.status_code} -> {res.json()}")

    banner("DEMONSTRATION COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()

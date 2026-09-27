import os
import time
import json
import hashlib
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Augmentation helpers (Fast CPU numpy transforms)
# ---------------------------------------------------------------------------

def random_hflip(arr):
    if random.random() < 0.5:
        return arr[:, ::-1, :].copy()
    return arr

def random_vflip(arr):
    if random.random() < 0.5:
        return arr[::-1, :, :].copy()
    return arr

def random_rot90(arr):
    k = random.randint(0, 3)
    return np.rot90(arr, k).copy()

def augment_sample(arr):
    """Apply spatial augmentations to (H, W, C) float32 array in [0, 1]."""
    arr = random_hflip(arr)
    arr = random_vflip(arr)
    arr = random_rot90(arr)
    return arr


# ---------------------------------------------------------------------------
# Data loading with Stratified 80/20 Train/Validation Split
# ---------------------------------------------------------------------------

def load_and_split_candidate_data(dir_path='candidate_tiles', train_ratio=0.8, augment_factor=2, seed=42):
    """
    Loads candidate tiles, performs a stratified 80% train / 20% validation split,
    and applies data augmentation exclusively to the training split.
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_by_class = {c: [] for c in CLASSES}
    for c in CLASSES:
        folder = os.path.join(dir_path, c)
        if not os.path.exists(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg')))
        for fname in files:
            fpath = os.path.join(folder, fname)
            img = Image.open(fpath).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0, 1]
            raw_by_class[c].append((arr, fname))

    train_X, train_y, train_fnames = [], [], []
    val_X, val_y, val_fnames = [], [], []

    for c in CLASSES:
        items = raw_by_class[c]
        random.shuffle(items)
        n_train = int(len(items) * train_ratio)
        train_items = items[:n_train]
        val_items = items[n_train:]

        label_idx = CLASS_TO_IDX[c]

        # Training set: original + augmented
        for arr, fname in train_items:
            train_X.append(arr.transpose(2, 0, 1))  # CHW
            train_y.append(label_idx)
            train_fnames.append(fname)
            for _ in range(augment_factor):
                aug_arr = augment_sample(arr)
                train_X.append(aug_arr.transpose(2, 0, 1))
                train_y.append(label_idx)
                train_fnames.append(f"aug_{fname}")

        # Validation set: clean original only
        for arr, fname in val_items:
            val_X.append(arr.transpose(2, 0, 1))
            val_y.append(label_idx)
            val_fnames.append(fname)

    return (
        torch.tensor(np.array(train_X), dtype=torch.float32),
        torch.tensor(np.array(train_y), dtype=torch.long),
        train_fnames,
        torch.tensor(np.array(val_X), dtype=torch.float32),
        torch.tensor(np.array(val_y), dtype=torch.long),
        val_fnames
    )


def load_eval(csv_path='eval_labels.csv', eval_dir='eval_set'):
    """Load independent evaluation set from ground-truth CSV."""
    df = pd.read_csv(csv_path)
    X, y, fnames = [], [], []
    for _, row in df.iterrows():
        fname = row['filename']
        label = row['true_label']
        fpath = os.path.join(eval_dir, fname)
        img = Image.open(fpath).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        X.append(arr.transpose(2, 0, 1))
        y.append(CLASS_TO_IDX[label])
        fnames.append(fname)
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.long),
        fnames
    )


# ---------------------------------------------------------------------------
# Model Architecture: Fast, High-Performance EuroSat CNN
# ---------------------------------------------------------------------------

class EuroSatCNN(nn.Module):
    """
    3-stage Convolutional Neural Network with BatchNorm, MaxPool, AdaptiveAvgPool,
    and Dropout regularization. Highly optimized for fast CPU training and sub-5ms inference.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 3 -> 32, 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 2: 32 -> 64, 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 3: 64 -> 128, 16x16 -> 1x1
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training & Evaluation Pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 65)
    print("GalaxEye EuroSAT CNN — Offline Training, Validation & Evaluation")
    print("=" * 65)

    print("\n[1/4] Loading & splitting candidate tiles (80% Train, 20% Val)...")
    X_train, y_train, train_fnames, X_val, y_val, val_fnames = load_and_split_candidate_data(
        dir_path='candidate_tiles',
        train_ratio=0.8,
        augment_factor=2,
        seed=42
    )
    print(f"    Train set:      {len(X_train)} samples (including 2× augmentation) across {len(CLASSES)} classes")
    print(f"    Validation set: {len(X_val)} samples (clean candidate held-out split)")

    print("\n[2/4] Loading independent evaluation set (eval_set)...")
    X_eval, y_eval, eval_fnames = load_eval()
    print(f"    Eval set:       {len(X_eval)} samples")

    model = EuroSatCNN(num_classes=len(CLASSES))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n    Model Architecture: EuroSatCNN — {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    epochs = 30
    batch_size = 64
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[3/4] Training for {epochs} epochs with validation checkpointing on CPU...\n")
    t0 = time.time()

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, X_train.size(0), batch_size):
            batch_idx = perm[i: i + batch_size]
            optimizer.zero_grad()
            out = model(X_train[batch_idx])
            loss = criterion(out, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        scheduler.step()

        # Evaluate on 20% Validation set after every epoch
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        marker = " * (Best Checkpoint)" if is_best else ""
        if (epoch + 1) % 5 == 0 or is_best or epoch == epochs - 1:
            print(f"    Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_loss/batches:.4f} | Val Acc: {val_acc*100:.1f}%{marker}")

    train_time = time.time() - t0
    print(f"\n    Training complete in {train_time:.1f}s.")
    print(f"    Selected best checkpoint from Epoch {best_epoch} with Val Acc: {best_val_acc*100:.2f}%")

    # ---------------------------------------------------------------------------
    # Final Evaluation on Independent Eval Set
    # ---------------------------------------------------------------------------
    print("\n[4/4] Evaluating Best Checkpoint on independent eval_set...")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        t_eval0 = time.time()
        logits = model(X_eval)
        eval_batch_time = time.time() - t_eval0
        eval_latency_ms = (eval_batch_time / len(X_eval)) * 1000
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

    y_true_np = y_eval.numpy()
    y_pred_np = preds.numpy()

    acc = (y_pred_np == y_true_np).mean()
    print(f"\n{'='*65}")
    print(f"  Final Evaluation Accuracy: {acc*100:.2f}% ({int(np.sum(y_pred_np == y_true_np))}/{len(y_true_np)})")
    print(f"  Batch CPU Inference Latency: {eval_latency_ms:.2f} ms/tile")
    print(f"{'='*65}\n")

    print("Classification Report (per-class precision/recall/F1):")
    print(classification_report(y_true_np, y_pred_np, target_names=CLASSES, digits=3))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true_np, y_pred_np)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print(cm_df)

    # ---------------------------------------------------------------------------
    # Save model artifact and metadata
    # ---------------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'landuse_cnn_v1.pt')
    torch.save(model.state_dict(), model_path)

    # Compute SHA-256 checksum
    hasher = hashlib.sha256()
    with open(model_path, 'rb') as f:
        hasher.update(f.read())
    model_hash = hasher.hexdigest()

    # Benchmark single-tile CPU latency
    model.eval()
    latencies = []
    sample_tensor = X_eval[:1]
    with torch.no_grad():
        for _ in range(50):
            t = time.perf_counter()
            model(sample_tensor)
            latencies.append((time.perf_counter() - t) * 1000)
    median_latency = float(np.median(latencies))

    metadata = {
        "model_name": "landuse_cnn",
        "model_version": "1.0.0",
        "architecture": "EuroSatCNN",
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "input_size": [3, 64, 64],
        "normalization": "div_255",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "sha256_checksum": model_hash,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "validation_best_accuracy": float(round(best_val_acc, 4)),
        "eval_accuracy": float(round(acc, 4)),
        "avg_cpu_latency_ms": median_latency,
        "benchmark_note": f"Median single-tile CPU latency {median_latency:.2f} ms on 64x64 RGB tile.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    meta_path = os.path.join('models', 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model weights -> {model_path}")
    print(f"Saved metadata     -> {meta_path}")
    print(f"Model SHA-256      : {model_hash}")
    print(f"Median single-tile CPU latency: {median_latency:.2f} ms")
    print(f"\n[DONE] Training & Evaluation Complete! Final Accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    train_and_evaluate()

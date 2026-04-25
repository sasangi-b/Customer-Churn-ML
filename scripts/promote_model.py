"""
promote_model.py
----------------
Run this BEFORE building the Docker image.
Finds the best model in mlruns/ by a metric and copies it to models/best_model/
so the Dockerfile can COPY it into the container at /app/model.

Usage:
    python scripts/promote_model.py
"""

import os
import glob
import shutil

METRIC      = "roc_auc"          # change to your metric
OUTPUT_DIR  = "models/best_model"

# ── Find all MLmodel files ────────────────────────────────────────────────────
patterns = [
    "./mlruns/*/models/*/artifacts",
    "./mlruns/*/*/artifacts/model",
]
candidates = []
for p in patterns:
    candidates.extend(glob.glob(p))

if not candidates:
    raise FileNotFoundError("No models found in mlruns/. Run the training pipeline first.")

# ── Pick the most recently modified model ─────────────────────────────────────
best = max(candidates, key=os.path.getmtime)
print(f"✅ Best model found at: {best}")

# ── Copy to models/best_model ─────────────────────────────────────────────────
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
shutil.copytree(best, OUTPUT_DIR)
print(f"✅ Model copied to: {OUTPUT_DIR}")

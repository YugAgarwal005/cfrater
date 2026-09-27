"""
Model Evaluation & Analysis
============================
Comprehensive evaluation of trained models with plots and reports.

Usage:
    python src/models/evaluate.py --model-dir models/ --data data/raw/problems.jsonl
"""

import argparse
import json
import logging
import sys
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RATING_BUCKETS = [
    800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    2400, 2500, 2600, 2700, 2800, 2900, 3000,
]


def snap_to_bucket(rating):
    r = int(round(float(rating) / 100.0)) * 100
    r = max(800, min(r, 3000))
    return min(RATING_BUCKETS, key=lambda b: abs(b - r))


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("rating"):
                    records.append(r)
            except Exception:
                pass
    return records


def print_distribution(records):
    """Print rating distribution table."""
    ratings = [r["rating"] for r in records]
    buckets = [snap_to_bucket(r) for r in ratings]
    count = Counter(buckets)

    print("\n📊 Rating Distribution in Dataset")
    print("=" * 50)
    print(f"{'Rating':>8} | {'Count':>6} | {'Bar'}")
    print("-" * 50)
    total = len(records)
    for bucket in RATING_BUCKETS:
        n = count.get(bucket, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>5}  | {n:>6} | {bar} {pct:.1f}%")
    print(f"\nTotal: {total:,} problems")


def evaluate_all_models(model_dir: str, data_path: str):
    """Load all models and run evaluation."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.feature_engineering.features import FeaturePipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
    import joblib

    model_dir = Path(model_dir)
    records = load_jsonl(data_path)
    print_distribution(records)

    ratings = [r["rating"] for r in records]
    _, val_idx = train_test_split(
        range(len(records)), test_size=0.15, random_state=42,
        stratify=[snap_to_bucket(r) for r in ratings]
    )
    val_records = [records[i] for i in val_idx]

    print(f"\n🔍 Evaluating on {len(val_records)} validation problems...")

    # Load pipeline
    pipeline = FeaturePipeline.load(str(model_dir / "feature_pipeline"))
    val_feats = pipeline.transform_batch(val_records)
    X_val = pipeline.build_feature_matrix(val_feats)

    y_reg_val = np.array([r["rating"] for r in val_records], dtype=np.float32)
    y_cls_val = np.array([RATING_BUCKETS.index(snap_to_bucket(r["rating"])) for r in val_records])

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    model_files = list(model_dir.glob("*.pkl"))
    for mf in sorted(model_files):
        if mf.name in ("best_regressor.pkl", "best_classifier.pkl"):
            continue
        try:
            model = joblib.load(mf)
            name = mf.stem

            if "regress" in name:
                preds = model.predict(X_val)
                mae = mean_absolute_error(y_reg_val, preds)
                rmse = math.sqrt(mean_squared_error(y_reg_val, preds))
                r2 = r2_score(y_reg_val, preds)
                w100 = np.mean(np.abs(preds - y_reg_val) <= 100) * 100
                w200 = np.mean(np.abs(preds - y_reg_val) <= 200) * 100
                w300 = np.mean(np.abs(preds - y_reg_val) <= 300) * 100
                print(f"\n📈 {name}")
                print(f"   MAE    = {mae:.1f}  (lower is better)")
                print(f"   RMSE   = {rmse:.1f}")
                print(f"   R²     = {r2:.4f}")
                print(f"   ±100   = {w100:.1f}%  ±200 = {w200:.1f}%  ±300 = {w300:.1f}%")

            elif "classif" in name:
                preds = model.predict(X_val)
                preds_rating = np.array([RATING_BUCKETS[min(max(int(p), 0), len(RATING_BUCKETS)-1)] for p in preds])
                true_ratings = y_reg_val
                acc = accuracy_score(y_cls_val, preds)
                adj = np.mean(np.abs(preds - y_cls_val) <= 1) * 100
                mae = mean_absolute_error(true_ratings, preds_rating)
                print(f"\n🎯 {name}")
                print(f"   Accuracy      = {acc*100:.1f}%")
                print(f"   Adj.Accuracy  = {adj:.1f}%  (within 1 bucket)")
                print(f"   MAE (ratings) = {mae:.1f}")

        except Exception as e:
            log.warning(f"Skipped {mf.name}: {e}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--data", default="data/raw/problems.jsonl")
    args = parser.parse_args()
    evaluate_all_models(args.model_dir, args.data)


if __name__ == "__main__":
    main()

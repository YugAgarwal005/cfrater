"""
Model Training Script — v3 with Data Augmentation
===================================================
Trains multiple models on Codeforces problem features.

Key improvements over v2:
  - Data augmentation: synthetically creates 3 copy-paste variants of every problem
      1. Original (LaTeX raw / Codeforces markdown)
      2. "Rendered" variant (strip LaTeX, normalize symbols)
      3. "Mangled" variant (simulate superscript-loss bug: 10^5 → 105)
    This triples the dataset and makes the model robust to all copy formats.
  - Isotonic regression calibration to fix rating range skew
  - Sample weights for rare ratings (extreme high/low)
  - Stacking meta-learner (Ridge on top of base regressors)
  - Improved XGBoost/LGBM hyperparameters with early stopping

Usage:
    python src/models/train.py
    python src/models/train.py --data data/processed/codeforces_full.jsonl --output models/
"""

import argparse
import copy
import json
import logging
import os
import sys
import math
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.feature_engineering.features import FeaturePipeline, normalize_cf_text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Rating buckets
# ---------------------------------------------------------------------------
RATING_BUCKETS = [
    800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    2400, 2500, 2600, 2700, 2800, 2900, 3000,
]


def snap_to_bucket(rating: int) -> int:
    if rating >= 3000:
        return 3000
    if rating <= 800:
        return 800
    return min(RATING_BUCKETS, key=lambda b: abs(b - rating))


def rating_to_class_idx(rating: int) -> int:
    return RATING_BUCKETS.index(snap_to_bucket(rating))


def class_idx_to_rating(idx: int) -> int:
    return RATING_BUCKETS[min(max(idx, 0), len(RATING_BUCKETS) - 1)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, min_rating: int = 800, max_rating: int = 3500) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                rating = rec.get("rating")
                if rating and min_rating <= rating <= max_rating:
                    records.append(rec)
            except Exception:
                pass
    log.info(f"Loaded {len(records):,} records from {path}")
    return records


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def _make_rendered_variant(record: dict) -> dict:
    """
    Create a 'rendered' variant of a problem record — simulating how the
    text looks when copy-pasted from a browser (MathJax rendered).
    
    Transforms:
      "*bold*"         → "bold"
      "$n \\leq 10^5$" → "n <= 10^5"
      "\\cdot"         → "*"
    """
    rec = copy.deepcopy(record)
    for field in ["statement", "input_spec", "output_spec", "note"]:
        t = rec.get(field) or ""
        if t:
            rec[field] = normalize_cf_text(t)
    return rec


def _make_mangled_variant(record: dict) -> dict:
    """
    Create a 'mangled' variant simulating the superscript-loss copy bug.
    
    When users copy from some PDF viewers or older browsers, the exponent
    in "10^5" gets merged into "105". This variant trains the model to
    still recognize such constraints correctly.
    
    Transforms:
      "10^5"  → "105"
      "10^9"  → "109"
      "10^18" → "1018"
      "n <= 2*10^5" → "n <= 2*105"
    """
    rec = copy.deepcopy(record)

    def mangle_text(t: str) -> str:
        if not t:
            return t
        # First normalize LaTeX forms to "10^N" so we can mangle them
        t = normalize_cf_text(t)
        # Now simulate superscript loss: 10^N → 10N
        t = re.sub(r'10\^(\d+)', lambda m: f'10{m.group(1)}', t)
        # Also mangle "2*10^5" → "2*105" → "2105" (some copiers merge everything)
        # Keep it mild — just drop the caret
        return t

    for field in ["statement", "input_spec", "output_spec", "note"]:
        t = rec.get(field) or ""
        if t:
            rec[field] = mangle_text(t)
    return rec


def _make_plaintext_variant(record: dict) -> dict:
    """
    Create a 'plain text' variant — simulating copy-paste without any formatting,
    preserving the raw LaTeX but with extra whitespace and line breaks removed.
    """
    rec = copy.deepcopy(record)
    for field in ["statement", "input_spec", "output_spec", "note"]:
        t = rec.get(field) or ""
        if t:
            # Strip markdown bold/italic but keep LaTeX
            t = re.sub(r'\*\*([^*]+)\*\*', r'\1', t)
            t = re.sub(r'\*([^*\n]+)\*', r'\1', t)
            # Normalize whitespace
            t = re.sub(r'\s+', ' ', t).strip()
            rec[field] = t
    return rec


def augment_records(records: list, augment_factor: int = 3) -> list:
    """
    Create augmented training records by generating multiple copy-paste variants.
    
    augment_factor:
      1 = original only
      2 = original + rendered
      3 = original + rendered + mangled
      4 = original + rendered + mangled + plain-text
    
    All augmented records keep the same rating label.
    """
    augmented = []
    for rec in records:
        augmented.append(rec)  # always include original
        if augment_factor >= 2:
            augmented.append(_make_rendered_variant(rec))
        if augment_factor >= 3:
            augmented.append(_make_mangled_variant(rec))
        if augment_factor >= 4:
            augmented.append(_make_plaintext_variant(rec))
    log.info(f"Augmented {len(records):,} → {len(augmented):,} records (factor {augment_factor}x)")
    return augmented


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(ratings: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights to upweight rare rating buckets (extremes).
    Without this, the model ignores 800-1000 and 2700+ problems.
    """
    from collections import Counter
    bucket_ratings = [snap_to_bucket(int(r)) for r in ratings]
    counts = Counter(bucket_ratings)
    max_count = max(counts.values())

    weights = []
    for r in bucket_ratings:
        freq_weight = max_count / counts[r]
        if r <= 1000:
            extreme_bonus = 3.0
        elif r <= 1200:
            extreme_bonus = 2.0
        elif r >= 2800:
            extreme_bonus = 2.5
        elif r >= 2500:
            extreme_bonus = 1.8
        elif r >= 2200:
            extreme_bonus = 1.3
        else:
            extreme_bonus = 1.0
        weights.append(freq_weight * extreme_bonus)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return weights


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred, label: str = ""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    within_100 = np.mean(np.abs(y_pred - y_true) <= 100) * 100
    within_200 = np.mean(np.abs(y_pred - y_true) <= 200) * 100
    within_300 = np.mean(np.abs(y_pred - y_true) <= 300) * 100
    log.info(
        f"[{label}] MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}  "
        f"±100={within_100:.1f}%  ±200={within_200:.1f}%  ±300={within_300:.1f}%"
    )
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "within_100": within_100, "within_200": within_200, "within_300": within_300,
    }


def evaluate_classification(y_true, y_pred, label: str = ""):
    acc = accuracy_score(y_true, y_pred)
    adj_acc = np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1) * 100
    log.info(f"[{label}] Accuracy={acc*100:.1f}%  Adjacent-1-bucket={adj_acc:.1f}%")
    return {"accuracy": acc, "adjacent_accuracy": adj_acc}


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(
    X_train, y_reg_train, y_cls_train,
    X_val, y_reg_val, y_cls_val,
    sample_weights, output_dir: Path,
) -> tuple:
    results = {}
    trained_models = {}

    # ── 1. Ridge Regression ──────────────────────────────────────────────
    log.info("Training Ridge Regression …")
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_reg_train, sample_weight=sample_weights)
    y_pred = ridge.predict(X_val)
    results["ridge_regression"] = evaluate_regression(y_reg_val, y_pred, "Ridge Regression")
    joblib.dump(ridge, output_dir / "ridge_regression.pkl")
    trained_models["ridge_regression"] = (ridge, y_pred)

    # ── 2. Random Forest Regressor ────────────────────────────────────────
    log.info("Training Random Forest Regressor …")
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=22, min_samples_leaf=1,
        max_features=0.4, n_jobs=-1, random_state=42,
    )
    rf_reg.fit(X_train, y_reg_train, sample_weight=sample_weights)
    y_pred = rf_reg.predict(X_val)
    results["rf_regression"] = evaluate_regression(y_reg_val, y_pred, "RF Regression")
    joblib.dump(rf_reg, output_dir / "rf_regression.pkl")
    trained_models["rf_regression"] = (rf_reg, y_pred)

    # ── 3. Random Forest Classifier ──────────────────────────────────────
    log.info("Training Random Forest Classifier …")
    rf_cls = RandomForestClassifier(
        n_estimators=150, max_depth=20, min_samples_leaf=2,
        max_features=0.4, n_jobs=-1, random_state=42,
    )
    rf_cls.fit(X_train, y_cls_train, sample_weight=sample_weights)
    y_pred_cls = rf_cls.predict(X_val)
    results["rf_classification"] = evaluate_classification(y_cls_val, y_pred_cls, "RF Classifier")
    joblib.dump(rf_cls, output_dir / "rf_classifier.pkl")

    # ── 4. XGBoost ────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        log.info("Training XGBoost Regressor …")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.55,
            reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
            tree_method="hist", device="cpu",
            n_jobs=-1, random_state=42,
            early_stopping_rounds=40,
        )
        xgb_reg.fit(
            X_train, y_reg_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_reg_val)],
            verbose=False,
        )
        y_pred = xgb_reg.predict(X_val)
        results["xgb_regression"] = evaluate_regression(y_reg_val, y_pred, "XGB Regression")
        joblib.dump(xgb_reg, output_dir / "xgb_regression.pkl")
        trained_models["xgb_regression"] = (xgb_reg, y_pred)

        log.info("Training XGBoost Classifier …")
        xgb_cls = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.06,
            subsample=0.75, colsample_bytree=0.55,
            reg_alpha=0.5, reg_lambda=1.0,
            tree_method="hist", device="cpu",
            n_jobs=-1, random_state=42,
            num_class=len(RATING_BUCKETS),
            objective="multi:softprob", eval_metric="mlogloss",
            early_stopping_rounds=20,
        )
        xgb_cls.fit(
            X_train, y_cls_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_cls_val)],
            verbose=False,
        )
        y_pred_cls = xgb_cls.predict(X_val)
        results["xgb_classification"] = evaluate_classification(y_cls_val, y_pred_cls, "XGB Classifier")
        joblib.dump(xgb_cls, output_dir / "xgb_classifier.pkl")

    # ── 5. LightGBM ──────────────────────────────────────────────────────
    if LGB_AVAILABLE:
        log.info("Training LightGBM Regressor …")
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.03,
            num_leaves=80, subsample=0.75, colsample_bytree=0.55,
            min_child_samples=5, reg_alpha=1.0, reg_lambda=2.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
        lgb_reg.fit(
            X_train, y_reg_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_reg_val)],
            callbacks=callbacks,
        )
        y_pred = lgb_reg.predict(X_val)
        results["lgb_regression"] = evaluate_regression(y_reg_val, y_pred, "LGB Regression")
        joblib.dump(lgb_reg, output_dir / "lgb_regression.pkl")
        trained_models["lgb_regression"] = (lgb_reg, y_pred)

    return results, trained_models


def train_stacking_ensemble(
    trained_models, X_val, y_reg_val, output_dir: Path,
) -> dict:
    reg_model_names = [k for k in trained_models if "regression" in k]
    if len(reg_model_names) < 2:
        return {}

    log.info(f"Training stacking meta-learner on {reg_model_names} …")
    val_preds = np.column_stack([
        trained_models[name][1] for name in reg_model_names
    ])

    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(val_preds, y_reg_val)
    meta_pred = meta.predict(val_preds)
    meta_results = evaluate_regression(y_reg_val, meta_pred, "Stacking Meta (val)")

    joblib.dump(meta, output_dir / "stacking_meta.pkl")
    joblib.dump(reg_model_names, output_dir / "stacking_model_names.pkl")
    log.info(f"Meta-learner weights: {dict(zip(reg_model_names, meta.coef_))}")
    return {"stacking": meta_results}


def train_calibrator(trained_models, X_val, y_reg_val, output_dir: Path):
    log.info("Fitting isotonic calibrator …")
    reg_names = [k for k in trained_models if "regression" in k]
    all_preds = np.column_stack([trained_models[name][1] for name in reg_names])
    ensemble_preds = all_preds.mean(axis=1)

    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(ensemble_preds, y_reg_val)

    cal_preds = calibrator.predict(ensemble_preds)
    cal_results = evaluate_regression(y_reg_val, cal_preds, "Calibrated Ensemble (val)")

    joblib.dump(calibrator, output_dir / "calibrator.pkl")
    log.info("Isotonic calibrator saved.")
    return calibrator, cal_results


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------

def pick_best_model(results: dict, output_dir: Path) -> dict:
    reg_models = {k: v for k, v in results.items() if "regression" in k}
    cls_models = {k: v for k, v in results.items() if "classification" in k}

    best_reg = min(reg_models, key=lambda k: reg_models[k]["mae"]) if reg_models else None
    best_cls = max(cls_models, key=lambda k: cls_models[k]["accuracy"]) if cls_models else None

    if best_reg:
        log.info(f"Best regression: {best_reg} MAE={reg_models[best_reg]['mae']:.1f}")
    if best_cls:
        log.info(f"Best classifier: {best_cls} Acc={cls_models[best_cls]['accuracy']*100:.1f}%")

    import shutil
    if best_reg:
        src = output_dir / f"{best_reg}.pkl"
        if src.exists():
            shutil.copy2(src, output_dir / "best_regressor.pkl")
    if best_cls:
        src = output_dir / f"{best_cls}.pkl"
        if src.exists():
            shutil.copy2(src, output_dir / "best_classifier.pkl")

    return {"best_regressor": best_reg, "best_classifier": best_cls, "all_results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/codeforces_full.jsonl")
    parser.add_argument("--output", default="models/")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--no-embeddings", action="store_true", default=True)
    parser.add_argument("--use-embeddings", action="store_true")
    parser.add_argument("--tfidf-features", type=int, default=18000)
    parser.add_argument("--min-rating", type=int, default=800)
    parser.add_argument("--max-rating", type=int, default=3500)
    parser.add_argument("--augment", type=int, default=4,
                        help="Data augmentation factor: 1=none, 2=+rendered, 3=+mangled, 4=+plaintext")
    args = parser.parse_args()

    use_emb = args.use_embeddings and not args.no_embeddings
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    records = load_data(args.data, args.min_rating, args.max_rating)
    if len(records) < 50:
        log.error("Not enough data. Run the scraper first.")
        sys.exit(1)

    ratings = [r["rating"] for r in records]
    log.info(f"Rating distribution: min={min(ratings)} max={max(ratings)} mean={np.mean(ratings):.0f}")

    # ── Train / val split BEFORE augmentation ─────────────────────────────
    # IMPORTANT: split on original records only, THEN augment training set.
    # This prevents the augmented versions from leaking into the val set.
    train_idx, val_idx = train_test_split(
        range(len(records)), test_size=args.test_size, random_state=42,
        stratify=[snap_to_bucket(r) for r in ratings]
    )
    train_records_orig = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]  # val stays as original

    # ── Augment training set ──────────────────────────────────────────────
    train_records = augment_records(train_records_orig, augment_factor=args.augment)

    # ── Feature pipeline ──────────────────────────────────────────────────
    # Fit on augmented training data so TF-IDF vocabulary covers all formats
    pipeline = FeaturePipeline(
        tfidf_max_features=args.tfidf_features,
        use_embeddings=use_emb,
    )
    pipeline.fit(train_records)
    pipeline.save(str(output_dir / "feature_pipeline"))

    log.info("Transforming training data …")
    train_features = pipeline.transform_batch(train_records)
    X_train = pipeline.build_feature_matrix(train_features)

    log.info("Transforming validation data …")
    val_features = pipeline.transform_batch(val_records)
    X_val = pipeline.build_feature_matrix(val_features)

    log.info(f"Feature matrix: train={X_train.shape}, val={X_val.shape}")

    # Labels
    y_reg_train = np.array([r["rating"] for r in train_records], dtype=np.float32)
    y_reg_val   = np.array([r["rating"] for r in val_records],   dtype=np.float32)
    y_cls_train = np.array([rating_to_class_idx(r["rating"]) for r in train_records])
    y_cls_val   = np.array([rating_to_class_idx(r["rating"]) for r in val_records])

    # Sample weights (applied to augmented training set)
    sample_weights = compute_sample_weights(y_reg_train)
    log.info(f"Sample weights: min={sample_weights.min():.2f} max={sample_weights.max():.2f}")

    # Train base models
    log.info("Starting model training …")
    results, trained_models = train_models(
        X_train, y_reg_train, y_cls_train,
        X_val, y_reg_val, y_cls_val,
        sample_weights, output_dir,
    )

    # Stacking meta-learner
    stacking_results = train_stacking_ensemble(trained_models, X_val, y_reg_val, output_dir)
    results.update(stacking_results)

    # Isotonic calibrator
    _, cal_results = train_calibrator(trained_models, X_val, y_reg_val, output_dir)
    results["calibrated_ensemble"] = cal_results

    # Best model
    summary = pick_best_model(results, output_dir)
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Training complete. Models saved to {output_dir}")
    log.info("=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    for name, metrics in results.items():
        log.info(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()

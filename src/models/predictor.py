"""
Predictor — Inference Engine v2
=================================
Loads trained models and produces rating predictions with confidence estimates.
Uses isotonic calibration + stacking meta-learner for better accuracy.

Usage:
    from src.models.predictor import RatingPredictor
    predictor = RatingPredictor("models/")
    result = predictor.predict(
        statement="...",
        input_spec="...",
        tags=["dp", "greedy"],
        solution_code="// C++ ...",
    )
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

log = logging.getLogger(__name__)

RATING_BUCKETS = [
    800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    2400, 2500, 2600, 2700, 2800, 2900, 3000,
]


def snap_to_bucket(rating: float) -> int:
    r = int(round(rating / 100.0)) * 100
    r = max(800, min(r, 3000))
    closest = min(RATING_BUCKETS, key=lambda b: abs(b - r))
    return closest


class RatingPredictor:
    """
    Loads all trained models and runs ensemble prediction with calibration.
    """

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self._load_pipeline()
        self._load_models()

    def _load_pipeline(self):
        from src.feature_engineering.features import FeaturePipeline
        pipeline_dir = self.model_dir / "feature_pipeline"
        if not pipeline_dir.exists():
            raise FileNotFoundError(f"Feature pipeline not found at {pipeline_dir}")
        self.pipeline = FeaturePipeline.load(str(pipeline_dir))
        log.info("Feature pipeline loaded")

    def _load_models(self):
        self.models = {}
        model_files = {
            "xgb_regressor": "xgb_regression.pkl",
            "lgb_regressor": "lgb_regression.pkl",
            "rf_regressor": "rf_regression.pkl",
            "ridge_regressor": "ridge_regression.pkl",
            "xgb_classifier": "xgb_classifier.pkl",
            "rf_classifier": "rf_classifier.pkl",
        }

        for name, fname in model_files.items():
            path = self.model_dir / fname
            if path.exists():
                self.models[name] = joblib.load(path)
                log.info(f"Loaded {name}")
            else:
                log.debug(f"Model not found: {fname}")

        if not self.models:
            raise FileNotFoundError("No trained models found. Run train.py first.")

        # Load stacking meta-learner (optional)
        self.stacking_meta = None
        self.stacking_model_names = None
        meta_path = self.model_dir / "stacking_meta.pkl"
        names_path = self.model_dir / "stacking_model_names.pkl"
        if meta_path.exists() and names_path.exists():
            self.stacking_meta = joblib.load(meta_path)
            self.stacking_model_names = joblib.load(names_path)
            log.info(f"Stacking meta-learner loaded (models: {self.stacking_model_names})")

        # Load isotonic calibrator (optional)
        self.calibrator = None
        cal_path = self.model_dir / "calibrator.pkl"
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)
            log.info("Isotonic calibrator loaded")

        # Load training summary for context
        summary_path = self.model_dir / "training_summary.json"
        self.training_summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                self.training_summary = json.load(f)

    def _build_record(
        self,
        statement: str = "",
        input_spec: str = "",
        output_spec: str = "",
        note: str = "",
        name: str = "",
        tags: Optional[list] = None,
        time_limit: str = "2 seconds",
        memory_limit: str = "256 megabytes",
    ) -> dict:
        return {
            "name": name,
            "statement": statement,
            "input_spec": input_spec,
            "output_spec": output_spec,
            "note": note,
            "tags": tags or [],
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }

    def predict(
        self,
        statement: str = "",
        input_spec: str = "",
        output_spec: str = "",
        note: str = "",
        name: str = "",
        tags: Optional[list] = None,
        time_limit: str = "2 seconds",
        memory_limit: str = "256 megabytes",
        solution_code: Optional[str] = None,
        # Legacy params (ignored)
        problem_index: str = "C",
        solved_count: int = 0,
    ) -> dict:
        """
        Predict difficulty rating for a problem.

        Returns:
            {
                "predicted_rating": 1500,
                "predicted_bucket": 1500,
                "estimated_range": [1400, 1600],
                "confidence": 0.72,
                "confidence_label": "Medium",
                "model_predictions": {...},
                "disclaimer": "...",
            }
        """
        record = self._build_record(
            statement=statement,
            input_spec=input_spec,
            output_spec=output_spec,
            note=note,
            name=name,
            tags=tags,
            time_limit=time_limit,
            memory_limit=memory_limit,
        )

        # Build features
        feats = self.pipeline.transform_single(record, solution_code=solution_code)

        # Convert structured dict → ordered 2D numpy row (MUST align to pipeline's column order)
        structured_row = np.array(
            [[feats["structured"].get(col, 0.0) for col in self.pipeline._structured_cols]],
            dtype=np.float32,
        )
        structured_row = np.nan_to_num(structured_row, nan=0.0, posinf=0.0, neginf=0.0)

        X = self.pipeline.build_feature_matrix({
            "tfidf":      feats["tfidf"],
            "tags":       feats["tags"],
            "structured": structured_row,
            "embedding":  feats["embedding"],
        })

        model_preds = {}

        # Regression predictions
        reg_preds = []
        reg_preds_ordered = []  # for stacking (ordered by stacking_model_names)
        for name_key, model in self.models.items():
            if "regressor" in name_key:
                try:
                    pred = float(model.predict(X)[0])
                    model_preds[name_key] = round(pred)
                    reg_preds.append(pred)
                except Exception as e:
                    log.debug(f"Model {name_key} failed: {e}")

        # Classification predictions
        cls_preds_rating = []
        for name_key, model in self.models.items():
            if "classifier" in name_key:
                try:
                    idx = int(model.predict(X)[0])
                    rating = RATING_BUCKETS[min(max(idx, 0), len(RATING_BUCKETS) - 1)]
                    model_preds[name_key] = rating
                    cls_preds_rating.append(rating)

                    # Confidence from class probabilities
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)[0]
                        model_preds[f"{name_key}_top_prob"] = float(proba[idx])
                except Exception as e:
                    log.debug(f"Model {name_key} failed: {e}")

        if not reg_preds:
            return {"error": "All models failed to produce predictions"}

        # ── Stacking meta-learner (if available) ──────────────────────────
        if self.stacking_meta is not None and self.stacking_model_names is not None:
            try:
                # Get ordered predictions matching training order
                stacking_input = []
                for mname in self.stacking_model_names:
                    # Map from training name ("xgb_regression") to predictor key ("xgb_regressor")
                    key = mname.replace("_regression", "_regressor").replace("rf_regression", "rf_regressor")
                    val = model_preds.get(key) or model_preds.get(mname)
                    if val is None:
                        val = float(np.mean(reg_preds))
                    stacking_input.append(float(val))

                stacking_vec = np.array([stacking_input])
                stacking_pred = float(self.stacking_meta.predict(stacking_vec)[0])
                model_preds["stacking_meta"] = round(stacking_pred)
                # Use stacking as the primary signal, weighted heavily
                primary_pred = stacking_pred
            except Exception as e:
                log.debug(f"Stacking meta failed: {e}")
                primary_pred = float(np.mean(reg_preds))
        else:
            primary_pred = float(np.mean(reg_preds))

        # ── Isotonic calibration ──────────────────────────────────────────
        if self.calibrator is not None:
            try:
                calibrated = float(self.calibrator.predict([primary_pred])[0])
                model_preds["calibrated"] = round(calibrated)
                ensemble_mean = calibrated
            except Exception as e:
                log.debug(f"Calibrator failed: {e}")
                ensemble_mean = primary_pred
        else:
            ensemble_mean = primary_pred

        # Also compute std across all regressors for confidence
        ensemble_std = float(np.std(reg_preds)) if len(reg_preds) > 1 else 200.0

        predicted_rating = snap_to_bucket(ensemble_mean)

        # Confidence estimate based on std
        max_std = 400.0
        raw_conf = max(0.0, 1.0 - (ensemble_std / max_std))
        confidence = round(raw_conf * 100)

        # Apply class probability boost if available
        top_probs = [v for k, v in model_preds.items() if "top_prob" in k]
        if top_probs:
            avg_prob = np.mean(top_probs)
            confidence = round((raw_conf * 0.6 + avg_prob * 0.4) * 100)

        confidence = max(5, min(95, confidence))

        if confidence >= 75:
            conf_label = "High"
        elif confidence >= 50:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        # Estimated range: widened by std
        bucket_idx = RATING_BUCKETS.index(predicted_rating)
        spread = max(1, round(ensemble_std / 100))
        lo_idx = max(0, bucket_idx - spread)
        hi_idx = min(len(RATING_BUCKETS) - 1, bucket_idx + spread)
        estimated_range = [RATING_BUCKETS[lo_idx], RATING_BUCKETS[hi_idx]]

        return {
            "predicted_rating": predicted_rating,
            "predicted_continuous": round(ensemble_mean),
            "estimated_range": estimated_range,
            "confidence": confidence,
            "confidence_label": conf_label,
            "ensemble_std": round(ensemble_std),
            "model_predictions": {k: v for k, v in model_preds.items() if "top_prob" not in k},
            "tags_used": feats.get("detected_tags", tags or []),
            "disclaimer": (
                "⚠️ This is an estimated difficulty rating based on historical Codeforces data. "
                "It is NOT an official Codeforces rating. Actual difficulty may vary."
            ),
        }

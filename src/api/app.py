"""
Flask API Server
================
Serves the rating predictor via REST API.

Endpoints:
  POST /api/predict        — predict rating from problem details
  GET  /api/health         — health check
  GET  /api/model-info     — model training summary

Usage:
    python src/api/app.py
    python src/api/app.py --port 5000 --debug
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="../../web", static_url_path="")
CORS(app)

_predictor = None
_request_count = 0  # global counter for diagnostics


def no_cache(response):
    """Add no-cache headers to every API response."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def get_predictor():
    global _predictor
    if _predictor is None:
        model_dir = os.environ.get("MODEL_DIR", "models/")
        try:
            from src.models.predictor import RatingPredictor
            _predictor = RatingPredictor(model_dir)
            log.info("Predictor loaded successfully")
        except Exception as e:
            log.error(f"Failed to load predictor: {e}")
            _predictor = None
    return _predictor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    web_dir = Path(__file__).parent.parent.parent / "web"
    resp = make_response(send_from_directory(str(web_dir), "index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/api/health")
def health():
    predictor = get_predictor()
    model_dir = Path(os.environ.get("MODEL_DIR", "models/"))
    summary = {}
    summary_path = model_dir / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    best_mae = None
    if summary.get("all_results"):
        reg_results = {k: v for k, v in summary["all_results"].items()
                       if "regression" in k and isinstance(v, dict) and "mae" in v}
        if reg_results:
            best_mae = round(min(v["mae"] for v in reg_results.values()), 1)
    resp = jsonify({
        "status": "ok" if predictor else "no_model",
        "model_loaded": predictor is not None,
        "message": "Models loaded and ready" if predictor else "No trained models found. Run train.py first.",
        "best_mae": best_mae,
        "best_model": summary.get("best_regressor"),
    })
    return no_cache(resp)


@app.route("/api/model-info")
def model_info():
    model_dir = Path(os.environ.get("MODEL_DIR", "models/"))
    summary_path = model_dir / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return no_cache(jsonify(json.load(f)))
    return no_cache(jsonify({"error": "No training summary found"}))


@app.route("/api/predict", methods=["POST"])
def predict():
    global _request_count
    _request_count += 1
    req_id = _request_count
    t0 = time.time()

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return no_cache(jsonify({"error": "No JSON body provided"})), 400

        # Log what we received
        stmt_preview = (data.get("statement") or "")[:80].replace("\n", " ")
        tags_in = data.get("tags", [])
        log.info(f"[req#{req_id}] statement='{stmt_preview}' tags={tags_in}")

        predictor = get_predictor()
        if predictor is None:
            log.info(f"[req#{req_id}] No model — returning demo response")
            return no_cache(jsonify(demo_response()))

        # Merge constraints textarea into input_spec
        constraints_raw = str(data.get("constraints") or "")
        input_spec = str(data.get("input_spec") or "")
        if constraints_raw:
            input_spec = f"{input_spec}\n{constraints_raw}".strip()

        # Parse time/memory limits from constraints text
        tl_match = re.search(r"time\s*limit[:\s]*(\d+(?:\.\d+)?)\s*(second|sec)", constraints_raw, re.I)
        ml_match = re.search(r"memory\s*limit[:\s]*(\d+)\s*(mb|megabyte|mib)", constraints_raw, re.I)
        time_limit   = f"{tl_match.group(1)} seconds"   if tl_match else str(data.get("time_limit") or "2 seconds")
        memory_limit = f"{ml_match.group(1)} megabytes" if ml_match else str(data.get("memory_limit") or "256 megabytes")

        statement    = str(data.get("statement") or "")
        output_spec  = str(data.get("output_spec") or "")
        note         = str(data.get("note") or "")
        name         = str(data.get("name") or "")
        tags         = list(data.get("tags") or [])
        if not tags:
            from src.feature_engineering.features import auto_detect_tags
            tags = auto_detect_tags(f"{statement} {input_spec} {constraints_raw}")
            if tags:
                log.info(f"[req#{req_id}] Auto-detected tags: {tags}")
        solution     = data.get("solution_code") or None

        result = predictor.predict(
            statement=statement,
            input_spec=input_spec,
            output_spec=output_spec,
            note=note,
            name=name,
            tags=tags,
            time_limit=time_limit,
            memory_limit=memory_limit,
            solution_code=solution,
        )

        elapsed = round((time.time() - t0) * 1000)
        log.info(f"[req#{req_id}] → rating={result.get('predicted_rating')}  "
                 f"models={result.get('model_predictions')}  ({elapsed}ms)")

        return no_cache(jsonify(result))

    except Exception as e:
        log.exception(f"[req#{req_id}] Prediction failed")
        return no_cache(jsonify({"error": str(e)})), 500


@app.route("/api/demo")
def demo():
    return no_cache(jsonify(demo_response()))


def demo_response() -> dict:
    """Fixed demo response — used only when no model is loaded."""
    return {
        "predicted_rating": 1500,
        "predicted_continuous": 1537,
        "estimated_range": [1400, 1600],
        "confidence": 72,
        "confidence_label": "Medium",
        "ensemble_std": 134,
        "model_predictions": {
            "xgb_regressor": 1560,
            "lgb_regressor": 1520,
            "rf_regressor": 1490,
            "ridge_regressor": 1480,
            "xgb_classifier": 1500,
            "rf_classifier": 1600,
        },
        "disclaimer": (
            "⚠️ This is an estimated difficulty rating based on historical Codeforces data. "
            "It is NOT an official Codeforces rating. Actual difficulty may vary."
        ),
        "_demo_mode": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model-dir", default="models/")
    args = parser.parse_args()

    os.environ["MODEL_DIR"] = args.model_dir

    log.info(f"Starting Flask server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

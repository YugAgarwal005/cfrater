#!/usr/bin/env python
"""
Quick Predict CLI
==================
Run a one-off prediction from the command line without the web UI.

Usage:
    python predict.py --statement "Given an array of n integers..." --tags dp,greedy
    python predict.py --interactive
    python predict.py --demo
"""
import argparse
import json
import sys
from pathlib import Path


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         CF Rating Predictor — Quick Predict CLI              ║
║         ⚠️  Estimates only, not official CF ratings          ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_result(result: dict):
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    r = result.get("predicted_rating", "?")
    lo, hi = result.get("estimated_range", [r, r])
    conf = result.get("confidence", "?")
    conf_label = result.get("confidence_label", "")

    print(f"""
┌─ Prediction Result ──────────────────────────────────────┐
│                                                           │
│   Predicted Rating:  {str(r):>6}                          │
│   Estimated Range:   {str(lo):>6} – {str(hi):<6}                   │
│   Confidence:        {str(conf):>3}% ({conf_label})               │
│                                                           │
└───────────────────────────────────────────────────────────┘
""")

    models = result.get("model_predictions", {})
    if models:
        print("Model Ensemble:")
        for name, val in models.items():
            print(f"  {name:<30} → {val}")
        print()

    print(f"⚠️  {result.get('disclaimer', '')}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statement", default="")
    parser.add_argument("--input-spec", default="")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--solution", default="")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    banner()

    if args.demo:
        result = {
            "predicted_rating": 1500,
            "estimated_range": [1400, 1600],
            "confidence": 72,
            "confidence_label": "Medium",
            "model_predictions": {
                "xgb_regressor": 1560,
                "lgb_regressor": 1520,
                "rf_regressor": 1490,
                "xgb_classifier": 1500,
            },
            "disclaimer": "Estimated only — not an official Codeforces rating.",
        }
        print_result(result)
        return

    # Interactive mode
    if args.interactive:
        print("Interactive mode — enter problem details (Ctrl+C to quit)\n")
        statement = input("Problem statement (paste here):\n> ").strip()
        tags = input("Tags (comma-separated, e.g. dp,greedy): ").strip()
        solution = input("Solution code (optional, press Enter to skip): ").strip()
    else:
        statement = args.statement
        tags = args.tags
        solution = args.solution or None

    if not statement:
        print("❌ No problem statement provided.")
        sys.exit(1)

    # Load predictor
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from src.models.predictor import RatingPredictor
        predictor = RatingPredictor(args.model_dir)
        result = predictor.predict(
            statement=statement,
            input_spec=args.input_spec,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
            solution_code=solution or None,
        )
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("No trained model found. Run: python src/models/train.py --data data/raw/problems.jsonl")
        sys.exit(1)

    print_result(result)


if __name__ == "__main__":
    main()

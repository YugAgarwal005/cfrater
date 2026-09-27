"""
run.py — Master runner script
==============================
One-stop script for all operations.

Usage:
    python run.py collect                          # Collect data from Codeforces
    python run.py collect --max-problems 2000      # Collect first 2000 problems
    python run.py collect --no-statements          # Faster: tags/ratings only
    python run.py train                            # Train all models
    python run.py train --no-embeddings            # Faster: skip sentence embeddings
    python run.py evaluate                         # Evaluate trained models
    python run.py serve                            # Start web UI server
    python run.py serve --port 8080                # Custom port
    python run.py predict --interactive            # CLI predict
    python run.py predict --demo                   # Demo predict
    python run.py all                              # Full pipeline (collect → train → serve)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run(cmd: list, **kwargs):
    print(f"\n▶  {' '.join(cmd)}\n" + "─" * 60)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n❌ Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="CF Rating Predictor — Master Runner")
    subparsers = parser.add_subparsers(dest="command")

    # collect
    p_collect = subparsers.add_parser("collect", help="Collect Codeforces data")
    p_collect.add_argument("--output", default="data/raw/problems.jsonl")
    p_collect.add_argument("--max-problems", type=int, default=0)
    p_collect.add_argument("--delay", type=float, default=0.8)
    p_collect.add_argument("--no-statements", action="store_true")
    p_collect.add_argument("--resume", action="store_true")

    # train
    p_train = subparsers.add_parser("train", help="Train ML models")
    p_train.add_argument("--data", default="data/raw/problems.jsonl")
    p_train.add_argument("--output", default="models/")
    p_train.add_argument("--no-embeddings", action="store_true")
    p_train.add_argument("--tfidf-features", type=int, default=15000)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained models")
    p_eval.add_argument("--model-dir", default="models/")
    p_eval.add_argument("--data", default="data/raw/problems.jsonl")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start web server")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--model-dir", default="models/")
    p_serve.add_argument("--debug", action="store_true")

    # predict
    p_predict = subparsers.add_parser("predict", help="CLI prediction")
    p_predict.add_argument("--interactive", action="store_true")
    p_predict.add_argument("--demo", action="store_true")
    p_predict.add_argument("--statement", default="")
    p_predict.add_argument("--tags", default="")

    # all
    p_all = subparsers.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--max-problems", type=int, default=0)
    p_all.add_argument("--no-embeddings", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    py = sys.executable

    if args.command == "collect":
        cmd = [py, "src/data_collection/scraper.py", "--output", args.output]
        if args.max_problems:
            cmd += ["--max-problems", str(args.max_problems)]
        if args.no_statements:
            cmd.append("--no-statements")
        if args.resume:
            cmd.append("--resume")
        cmd += ["--delay", str(args.delay)]
        run(cmd)

    elif args.command == "train":
        cmd = [py, "src/models/train.py", "--data", args.data, "--output", args.output]
        if args.no_embeddings:
            cmd.append("--no-embeddings")
        cmd += ["--tfidf-features", str(args.tfidf_features)]
        run(cmd)

    elif args.command == "evaluate":
        cmd = [py, "src/models/evaluate.py", "--model-dir", args.model_dir, "--data", args.data]
        run(cmd)

    elif args.command == "serve":
        cmd = [py, "src/api/app.py", "--port", str(args.port), "--model-dir", args.model_dir]
        if args.debug:
            cmd.append("--debug")
        print(f"\n🌐 Open browser at: http://localhost:{args.port}\n")
        run(cmd)

    elif args.command == "predict":
        cmd = [py, "predict.py", "--model-dir", "models/"]
        if args.interactive:
            cmd.append("--interactive")
        elif args.demo:
            cmd.append("--demo")
        elif args.statement:
            cmd += ["--statement", args.statement, "--tags", args.tags]
        else:
            cmd.append("--demo")
        run(cmd)

    elif args.command == "all":
        print("🚀 Running full pipeline: collect → train → serve")

        # Step 1: Collect
        collect_cmd = [py, "src/data_collection/scraper.py", "--output", "data/raw/problems.jsonl"]
        if args.max_problems:
            collect_cmd += ["--max-problems", str(args.max_problems)]
        run(collect_cmd)

        # Step 2: Train
        train_cmd = [py, "src/models/train.py", "--data", "data/raw/problems.jsonl", "--output", "models/"]
        if args.no_embeddings:
            train_cmd.append("--no-embeddings")
        run(train_cmd)

        # Step 3: Serve
        serve_cmd = [py, "src/api/app.py", "--port", "5000", "--model-dir", "models/"]
        print("\n🌐 Open browser at: http://localhost:5000\n")
        run(serve_cmd)


if __name__ == "__main__":
    main()

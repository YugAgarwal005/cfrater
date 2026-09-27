"""
Dataset Analysis & EDA Helper
==============================
Quick analysis of the collected Codeforces dataset.

Usage:
    python notebooks/eda.py --data data/raw/problems.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter


def analyze(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("rating"):
                    records.append(r)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {path}")
    print(f"{'='*60}\n")
    print(f"Total problems: {len(records):,}\n")

    # Rating distribution
    ratings = [r["rating"] for r in records]
    print("Rating Distribution:")
    buckets = range(800, 3100, 100)
    for b in buckets:
        count = sum(1 for r in ratings if r == b)
        bar = "█" * min(count // 10, 40)
        if count:
            print(f"  {b:>5}: {count:>5}  {bar}")

    print(f"\n  Min: {min(ratings)}  Max: {max(ratings)}  Mean: {sum(ratings)/len(ratings):.0f}")

    # Tag distribution
    all_tags = []
    for r in records:
        all_tags.extend(r.get("tags", []))
    tag_counts = Counter(all_tags)
    print(f"\nTop 15 Tags:")
    for tag, count in tag_counts.most_common(15):
        print(f"  {count:>5}  {tag}")

    # Statement coverage
    has_stmt = sum(1 for r in records if r.get("statement", "").strip())
    print(f"\nStatement coverage: {has_stmt:,}/{len(records):,} ({has_stmt/len(records)*100:.1f}%)")

    # Index distribution
    indexes = Counter(r.get("index", "?")[0] for r in records)
    print(f"\nProblem index distribution: {dict(indexes)}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/problems.jsonl"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Run: python src/data_collection/scraper.py --output data/raw/problems.jsonl")
    else:
        analyze(data_path)

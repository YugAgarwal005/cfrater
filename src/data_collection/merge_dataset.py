"""
Merge Codeforces API rated problems with 7k problem statements dataset.
Produces data/processed/codeforces_full.jsonl with real statements and ratings.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main():
    log.info("Fetching latest problem metadata from Codeforces API...")
    resp = requests.get("https://codeforces.com/api/problemset.problems", timeout=20).json()
    if resp.get("status") != "OK":
        raise RuntimeError(f"Codeforces API error: {resp}")
    problems = resp["result"]["problems"]
    log.info(f"Fetched {len(problems):,} total problems from API")

    cf_lookup = {
        (p.get("contestId"), (p.get("name") or "").strip().lower()): p
        for p in problems if "rating" in p
    }
    cf_name_lookup = {
        (p.get("name") or "").strip().lower(): p
        for p in problems if "rating" in p
    }
    log.info(f"Indexed {len(cf_lookup):,} rated problems from CF API")

    parquet_path = Path("data/raw/codeforces_7k.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")

    log.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df):,} problem statements from parquet")

    merged = []
    seen_ids = set()

    for _, r in df.iterrows():
        cid = r.get("contestId")
        name_clean = (r.get("name") or "").strip().lower()
        p = cf_lookup.get((cid, name_clean)) or cf_name_lookup.get(name_clean)
        if not p:
            continue

        prob_id = f"{p.get('contestId')}_{p.get('index')}"
        if prob_id in seen_ids:
            continue
        seen_ids.add(prob_id)

        # Merge tags
        tags_set = set(p.get("tags", []))
        dataset_tags = r.get("tags")
        if isinstance(dataset_tags, list):
            tags_set.update(dataset_tags)
        elif isinstance(dataset_tags, str):
            try:
                tags_set.update(json.loads(dataset_tags.replace("'", '"')))
            except Exception:
                pass

        stmt = str(r.get("problem-description") or "").strip()
        inp = str(r.get("input-specification") or "").strip()
        outp = str(r.get("output-specification") or "").strip()
        note = str(r.get("note") or "").strip()

        if len(stmt) < 20:
            continue

        rec = {
            "problem_id": prob_id,
            "contest_id": p.get("contestId"),
            "index": p.get("index", "C"),
            "name": p.get("name", r.get("name", "")),
            "rating": int(p["rating"]),
            "tags": sorted(list(tags_set)),
            "statement": stmt,
            "input_spec": inp,
            "output_spec": outp,
            "note": note,
            "time_limit": str(r.get("time-limit") or "2.0 seconds"),
            "memory_limit": str(r.get("memory-limit") or "256 megabytes"),
        }
        merged.append(rec)

    log.info(f"Successfully matched and cleaned {len(merged):,} complete problem records!")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "codeforces_full.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

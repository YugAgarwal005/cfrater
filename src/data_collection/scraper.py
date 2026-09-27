"""
Codeforces Problem Scraper
==========================
Collects problem metadata, statements, and ratings from the Codeforces API.

Usage:
    python src/data_collection/scraper.py --output data/raw/problems.jsonl
    python src/data_collection/scraper.py --output data/raw/problems.jsonl --max-problems 5000
    python src/data_collection/scraper.py --resume --output data/raw/problems.jsonl
"""

import argparse
import json
import os
import sys
import time
import logging
import re
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CF_API_BASE = "https://codeforces.com/api"
CF_PROBLEM_URL = "https://codeforces.com/problemset/problem/{contest}/{index}"

RATING_CLASSES = [
    800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    2400, 2500, 2600, 2700, 2800, 2900, 3000,
]

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                   requests.exceptions.Timeout)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def _get(url: str, params: dict = None, timeout: int = 20) -> dict:
    """Perform a GET request with retries."""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_all_problems() -> list[dict]:
    """Fetch the full problem list from the Codeforces API."""
    log.info("Fetching full problem list from Codeforces API …")
    data = _get(f"{CF_API_BASE}/problemset.problems")
    if data["status"] != "OK":
        raise RuntimeError(f"API returned status: {data['status']}")
    problems = data["result"]["problems"]
    stats = {s["contestId"]: s for s in data["result"]["problemStatistics"]}
    log.info(f"  Got {len(problems):,} problems total")
    return problems, stats


def _html_to_text(html: str) -> str:
    """Convert HTML problem statement to plain text."""
    if not html:
        return ""
    if HTML2TEXT_AVAILABLE:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html).strip()
    # Fallback: strip tags with regex
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_problem_statement(contest_id: int, problem_index: str) -> dict:
    """
    Attempt to fetch a problem's HTML statement from Codeforces.
    Returns a dict with keys: statement, input_spec, output_spec, note.
    Rate-limited to avoid hammering the server.
    """
    url = CF_PROBLEM_URL.format(contest=contest_id, index=problem_index)
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; CF-Research-Bot/1.0)"
        })
        if resp.status_code != 200:
            return {}

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        result = {}

        # Problem statement
        stmt_div = soup.find("div", class_="problem-statement")
        if not stmt_div:
            return {}

        # Extract sections
        sections = {
            "statement": None,
            "input-specification": None,
            "output-specification": None,
            "note": None,
        }

        for div in stmt_div.find_all("div", recursive=False):
            classes = div.get("class", [])
            if "header" in classes:
                continue
            title_div = div.find("div", class_="section-title")
            if title_div:
                title = title_div.get_text().strip().lower()
                title_div.decompose()
                if "input" in title:
                    sections["input-specification"] = _html_to_text(str(div))
                elif "output" in title:
                    sections["output-specification"] = _html_to_text(str(div))
                elif "note" in title:
                    sections["note"] = _html_to_text(str(div))
            else:
                if sections["statement"] is None:
                    sections["statement"] = _html_to_text(str(div))

        result = {k: v or "" for k, v in sections.items()}

        # Time / memory limits
        tl_tag = stmt_div.find("div", class_="time-limit")
        ml_tag = stmt_div.find("div", class_="memory-limit")
        result["time_limit"] = tl_tag.get_text().replace("time limit per test", "").strip() if tl_tag else ""
        result["memory_limit"] = ml_tag.get_text().replace("memory limit per test", "").strip() if ml_tag else ""

        return result

    except Exception as exc:
        log.debug(f"Statement fetch failed for {contest_id}{problem_index}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(
    output_path: str,
    max_problems: int = 0,
    delay: float = 0.8,
    fetch_statements: bool = True,
    resume: bool = False,
):
    """
    Full collection pipeline:
    1. Fetch problem list via API
    2. Filter to rated problems only
    3. Optionally fetch HTML statements
    4. Write to JSONL
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-collected IDs
    collected_ids: set[str] = set()
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    collected_ids.add(f"{rec['contest_id']}_{rec['index']}")
                except Exception:
                    pass
        log.info(f"Resuming — {len(collected_ids):,} already collected")

    problems, stats = fetch_all_problems()

    # Filter: must have a rating
    rated = [p for p in problems if p.get("rating")]
    log.info(f"  Rated problems: {len(rated):,}")

    # Sort by contestId descending (newer first, usually more representative)
    rated.sort(key=lambda p: (p.get("contestId", 0), p.get("index", "")), reverse=True)

    if max_problems and max_problems > 0:
        rated = rated[:max_problems]
        log.info(f"  Limiting to {max_problems:,} problems")

    mode = "a" if resume else "w"
    skipped = 0
    written = 0

    with open(out_path, mode, encoding="utf-8") as f_out:
        bar = tqdm(rated, desc="Collecting problems", unit="prob")
        for prob in bar:
            contest_id = prob.get("contestId")
            index = prob.get("index", "")
            uid = f"{contest_id}_{index}"

            if uid in collected_ids:
                skipped += 1
                bar.set_postfix(written=written, skipped=skipped)
                continue

            record = {
                "problem_id": uid,
                "contest_id": contest_id,
                "index": index,
                "name": prob.get("name", ""),
                "rating": prob.get("rating"),
                "tags": prob.get("tags", []),
                "points": prob.get("points"),
                "solved_count": stats.get(contest_id, {}).get("solvedCount", 0)
                if isinstance(stats.get(contest_id), dict) else 0,
                "statement": "",
                "input_spec": "",
                "output_spec": "",
                "note": "",
                "time_limit": "",
                "memory_limit": "",
            }

            # Fetch statement HTML
            if fetch_statements and contest_id:
                stmt = fetch_problem_statement(contest_id, index)
                record["statement"] = stmt.get("statement", "")
                record["input_spec"] = stmt.get("input-specification", "")
                record["output_spec"] = stmt.get("output-specification", "")
                record["note"] = stmt.get("note", "")
                record["time_limit"] = stmt.get("time_limit", "")
                record["memory_limit"] = stmt.get("memory_limit", "")
                time.sleep(delay)

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            written += 1
            bar.set_postfix(written=written, rating=record["rating"])

    log.info(f"Done. Written {written:,} records to {out_path}")
    log.info(f"Skipped {skipped:,} already-collected records")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect Codeforces problems")
    parser.add_argument("--output", default="data/raw/problems.jsonl")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="Max problems to collect (0 = all)")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Seconds between statement fetches")
    parser.add_argument("--no-statements", action="store_true",
                        help="Skip fetching HTML statements (faster)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous collection run")
    args = parser.parse_args()

    collect(
        output_path=args.output,
        max_problems=args.max_problems,
        delay=args.delay,
        fetch_statements=not args.no_statements,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

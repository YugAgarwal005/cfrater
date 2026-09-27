"""
Feature Engineering Pipeline
==============================
Extracts rich features from Codeforces problem records for ML training.

Text Normalization
------------------
Codeforces problems can be copy-pasted in multiple ways, each producing different
text artifacts. We normalize all of them to a canonical form before extracting features:

  1. LaTeX copy (raw):    "1 \\leq n \\leq 2 \\cdot 10^5"   → "$1 \\leq n \\leq 2 \\cdot 10^5$"
  2. Browser text copy:  "1 ≤ n ≤ 2·10⁵"                  (Unicode math chars)
  3. Paste-as-text bug:  "1 ≤ n ≤ 2·105"                   (superscripts merged into base)

All three are normalized to the same canonical form before TF-IDF and constraint extraction.

Feature groups:
  1. text_tfidf         — TF-IDF on normalized statement + input/output spec
  2. embedding          — sentence-transformer dense embeddings
  3. tags               — multi-hot encoded topic tags
  4. constraints        — parsed numeric constraint bounds (from all copy formats)
  5. structural         — problem length, formula density, examples, etc.
  6. solution           — algorithm / data-structure keywords from C++ solution
  7. difficulty_signals — keyword-based difficulty tier features
"""

import re
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known Codeforces tags (union of most common)
# ---------------------------------------------------------------------------
ALL_TAGS = [
    "implementation", "math", "greedy", "dp", "data structures",
    "brute force", "constructive algorithms", "graphs", "sortings",
    "binary search", "dfs and similar", "trees", "strings", "number theory",
    "combinatorics", "special", "geometry", "bitmasks", "two pointers",
    "dsu", "shortest paths", "probabilities", "divide and conquer", "hashing",
    "games", "flows", "interactive", "matrices", "string suffix structures",
    "fft", "graph matchings", "ternary search", "expression parsing",
    "meet-in-the-middle", "2-sat", "chinese remainder theorem",
    "schedules", "seg tree", "bit manipulation",
]

# ---------------------------------------------------------------------------
# Unicode math → ASCII mapping
# ---------------------------------------------------------------------------
UNICODE_SUPERSCRIPT_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
}
UNICODE_SUBSCRIPT_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
}

# Powers of 10 that are "round" numbers from copy-paste superscript loss.
# e.g. "10^5" → "105" in pasted text. We map these back.
# Key: merged string, Value: actual value as string
MERGED_POWER_OF_10 = {
    "102": "100",        # 10^2
    "103": "1000",       # 10^3
    "104": "10000",      # 10^4
    "105": "100000",     # 10^5
    "106": "1000000",    # 10^6
    "107": "10000000",   # 10^7
    "108": "100000000",  # 10^8
    "109": "1000000000", # 10^9
    "1012": "1000000000000",  # 10^12
    "1018": "1000000000000000000",  # 10^18
}
# Compile regex for merged-power detection (word boundaries important)
_MERGED_POWER_RE = re.compile(
    r'\b(10(?:18|12|9|8|7|6|5|4|3|2))\b'
)


def _replace_merged_powers(text: str) -> str:
    """Replace merged power-of-ten artifacts like '105' → '100000'."""
    def _sub(m):
        key = m.group(1)
        return MERGED_POWER_OF_10.get(key, key)
    return _MERGED_POWER_RE.sub(_sub, text)


def _unicode_superscripts_to_caret(text: str) -> str:
    """Convert Unicode superscript digits to ^N notation, grouping multi-digit exponents."""
    # Replace each unicode superscript digit with a tagged form
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in UNICODE_SUPERSCRIPT_MAP:
            # Collect all consecutive superscript digits
            exp_digits = []
            while i < len(text) and text[i] in UNICODE_SUPERSCRIPT_MAP:
                exp_digits.append(UNICODE_SUPERSCRIPT_MAP[text[i]])
                i += 1
            result.append('^' + ''.join(exp_digits))
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def _unicode_subscripts_to_underscore(text: str) -> str:
    """Convert Unicode subscript digits to _N notation."""
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in UNICODE_SUBSCRIPT_MAP:
            sub_digits = []
            while i < len(text) and text[i] in UNICODE_SUBSCRIPT_MAP:
                sub_digits.append(UNICODE_SUBSCRIPT_MAP[text[i]])
                i += 1
            result.append('_' + ''.join(sub_digits))
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def normalize_cf_text(text: str) -> str:
    """
    Normalize Codeforces problem text that may have been copy-pasted in
    different ways (LaTeX raw, browser-rendered, or with superscript-loss bugs).

    All three copy modes are normalized to the same canonical ASCII form so that
    the TF-IDF and constraint extractor see consistent text regardless of how
    the user copy-pasted from Codeforces.

    Examples:
      "n \\leq 10^5"          → "n <= 10^5"
      "n ≤ 2·10⁵"             → "n <= 2*10^5"
      "n ≤ 105"  (bug copy)   → "n <= 100000"
    """
    if not text:
        return text

    # ── Step 1: Unicode superscripts → ^N ───────────────────────────────
    text = _unicode_superscripts_to_caret(text)
    # ── Step 2: Unicode subscripts → _N ─────────────────────────────────
    text = _unicode_subscripts_to_underscore(text)

    # ── Step 3: LaTeX inequality symbols → ASCII ─────────────────────────
    text = text.replace('\\leq', '<=').replace('\\le ', '<= ').replace('\\le\n', '<= \n')
    text = text.replace('\\geq', '>=').replace('\\ge ', '>= ').replace('\\ge\n', '>= \n')
    text = text.replace('≤', '<=').replace('≥', '>=')
    text = text.replace('⩽', '<=').replace('⩾', '>=')

    # ── Step 4: Multiplication symbols → * ───────────────────────────────
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    text = text.replace('×', '*').replace('·', '*').replace('⋅', '*')

    # ── Step 5: LaTeX braces in exponents: 10^{5} → 10^5 ────────────────
    text = re.sub(r'(\d+)\^\{(\d+)\}', r'\1^\2', text)
    text = re.sub(r'10\^\\{(\d+)\\}', r'10^\1', text)  # edge case

    # ── Step 6: LaTeX \frac{a}{b} → a/b ─────────────────────────────────
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)

    # ── Step 7: Strip dollar signs around math expressions ───────────────
    # $expr$ → expr (keep content, remove delimiters)
    text = re.sub(r'\$\$([^$]+)\$\$', r' \1 ', text)  # display math $$...$$
    text = re.sub(r'\$([^$\n]{1,80})\$', r' \1 ', text)  # inline math $...$

    # ── Step 8: Codeforces Markdown italics/bold: *word* → word ──────────
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)    # *italic*

    # ── Step 9: Detect and fix superscript-loss bug: 105 → 100000 ────────
    # Only apply AFTER other cleanup to avoid false positives
    # Heuristic: "10N" where N is 2-18 at a word boundary (not inside larger numbers)
    text = _replace_merged_powers(text)

    # ── Step 10: Normalize whitespace ────────────────────────────────────
    text = re.sub(r'[ \t]{2,}', ' ', text)  # multiple spaces → single
    text = re.sub(r'\n{3,}', '\n\n', text)  # triple newlines → double

    # ── Step 11: Other LaTeX cleanup ─────────────────────────────────────
    text = re.sub(r'\\(?:textbf|textit|text|mathrm|mathbf|mathit)\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\(?:left|right)[\[\](){}|]', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\b', ' ', text)  # remaining LaTeX commands → space

    return text.strip()


# ---------------------------------------------------------------------------
# Constraint patterns (now applied on normalized text)
# ---------------------------------------------------------------------------
# After normalize_cf_text(), all forms look the same:
#   "n <= 2*10^5", "1 <= n <= 100000", "n <= 100000"
# These patterns match all those forms.
CONSTRAINT_PATTERNS = {
    "n": [
        r"(?:^|[^a-z])n\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"1\s*<=\s*n\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"n\s+is\s+(?:at\s+most|up\s+to)\s+([0-9][0-9\s\*\^e\.]+)",
        r"(?:array|sequence|string)\s+of\s+(?:length\s+)?([0-9][0-9\s\*\^e\.]+)",
        r"n\s*=\s*([0-9][0-9\s\*\^e\.]+)",
    ],
    "q": [
        r"(?:^|[^a-z])q\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"1\s*<=\s*q\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"(?:queries|operations)\s+(?:of\s+)?([0-9][0-9\s\*\^e\.]+)",
    ],
    "k": [
        r"(?:^|[^a-z])k\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"1\s*<=\s*k\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
    ],
    "m": [
        r"(?:^|[^a-z])m\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"1\s*<=\s*m\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
    ],
    "t": [
        r"(?:^|[^a-z])t\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"1\s*<=\s*t\s*<=\s*([0-9][0-9\s\*\^e\.]+)",
        r"(?:test cases|testcases)\s+(?:up to|at most)\s+([0-9][0-9\s\*\^e\.]+)",
    ],
}

# Also look for known large round numbers that commonly appear as constraints
COMMON_CONSTRAINT_VALUES = [
    1e5, 2e5, 3e5, 5e5, 1e6, 2e6, 1e7, 1e8, 1e9, 2e9, 1e12, 1e15, 1e18
]


# ---------------------------------------------------------------------------
# Solution keyword categories
# ---------------------------------------------------------------------------
SOLUTION_KEYWORDS = {
    "segment_tree": ["segment_tree", "segtree", "fenwick", "bit_tree", "seg tree", "tree.*update.*query"],
    "binary_indexed_tree": ["fenwick", "BIT", "binary indexed"],
    "dsu": ["dsu", "union.find", "make_set", "find_set"],
    "dijkstra": ["dijkstra", "priority_queue.*dist", "dist.*priority_queue"],
    "bfs": [r"\bbfs\b", "breadth.*first", "queue.*push.*pop"],
    "dfs": [r"\bdfs\b", "depth.*first", "recursive.*visit"],
    "dp": [r"\bdp\b", "memoiz", "dynamic.programm", r"dp\["],
    "binary_search": ["lower_bound", "upper_bound", "binary.*search", r"\bbsearch\b"],
    "sorting": [r"sort\(", r"stable_sort", r"partial_sort", r"nth_element"],
    "hashing": ["unordered_map", "hash_map", "polynomial.*hash", "rolling.*hash"],
    "trie": ["trie", "prefix.*tree"],
    "sparse_table": ["sparse.*table", "rmq"],
    "lca": [r"\blca\b", "lowest.*common.*ancestor", "lift.*ancestor"],
    "matrix_exp": ["matrix.*mul", "mat_mul", "matrix.*pow"],
    "fft": [r"\bfft\b", "fast.*fourier", "ntt", "number.*theoretic"],
    "two_pointers": ["two.*pointer", "sliding.*window", "lo.*hi.*while"],
    "greedy": ["greedy", "sort.*then.*pick"],
    "string_algo": ["kmp", "z.function", "aho.*corasick", "suffix.*array", "manacher"],
    "game_theory": ["nim", "sprague.*grundy", "grundy", "mex"],
    "flow": ["max_flow", "min_cut", "dinic", "edmonds.*karp"],
    "geometry": ["cross.*product", "dot.*product", "convex.*hull", "polygon.*area"],
}

# Complexity keywords → estimated complexity bucket
COMPLEXITY_KEYWORDS = {
    "O(1)": [r"\bO\(1\)", r"\bo\(1\)"],
    "O(logN)": [r"\bO\(log", r"binary search.*O\(log"],
    "O(N)": [r"\bO\(n\b", r"\bO\(N\b", "linear time", r"O\(n\)"],
    "O(NlogN)": [r"O\(n log", r"O\(N log", r"O\(n\\log", "nlogn", "n log n"],
    "O(N^2)": [r"O\(n\^2\)", r"O\(N\^2\)", r"O\(n2\)", "quadratic"],
    "O(N^3)": [r"O\(n\^3\)", "cubic"],
    "O(2^N)": [r"O\(2\^n\)", "exponential", "bitmask.*all subsets"],
}

# ---------------------------------------------------------------------------
# Difficulty-tier correlated keyword groups
# ---------------------------------------------------------------------------
DIFFICULTY_KEYWORDS = {
    "easy_impl": [
        r"\b(output|print|find the|count the|check if|determine|compute)\b",
        r"\b(array|sequence|list|sum|maximum|minimum|even|odd)\b",
    ],
    "mid_algo": [
        r"\b(binary search|greedy|two pointer|sliding window|prefix sum|suffix sum)\b",
        r"\b(sorting|comparator|frequency|occurrence|deque|stack|queue)\b",
    ],
    "hard_ds": [
        r"\b(segment tree|fenwick|sparse table|merge sort tree|persistent|sqrt decomp)\b",
        r"\b(graph|strongly connected|bipartite|euler|hamiltonian|topological)\b",
        r"\b(dp on tree|tree dp|centroid decomp|heavy.light decomp|lca)\b",
    ],
    "expert_algo": [
        r"\b(FFT|NTT|polynomial|generating function|fast fourier)\b",
        r"\b(network flow|min cut|max flow|matching|assignment problem)\b",
        r"\b(suffix array|suffix automaton|aho.corasick|palindrome tree|eertree)\b",
        r"\b(2-sat|linear programming|hall.*theorem|dilworth|ramsey)\b",
        r"\b(matroid|algebraic|galois|group theory|burnside|polya)\b",
        r"\b(offline.*query|mo.*algorithm|sqrt.*block|heavy.light)\b",
    ],
}

# ---------------------------------------------------------------------------
# Auto-tag regex detection rules
# ---------------------------------------------------------------------------
TAG_DETECTION_RULES = [
    ("dp", r"\b(dynamic.programm|memoiz|dp\b|knapsack|subproblem|optimal substructure)\b"),
    ("graphs", r"\b(graph|node|edge|vertex|vertices|adjacen|connected component|bipartite|dag\b)\b"),
    ("trees", r"\b(rooted.tree|binary tree|parent.*child|subtree|forest|lca|ancestor|leaf node)\b"),
    ("binary search", r"\b(binary.search|bisect|lower.bound|upper.bound|monoton)\b"),
    ("greedy", r"\b(greedy|always.pick|maximum.*first|minimum.*first|exchange argument)\b"),
    ("math", r"\b(prime|modulo|factorial|gcd|lcm|fibonacci|euler|totient|combinatorics|pigeonhole|divisib)\b"),
    ("number theory", r"\b(prime|divisor|gcd|lcm|euler|totient|sieve|modular inverse|coprime)\b"),
    ("sortings", r"\b(sort|sorted|ascending|descending|order.*element|arrange)\b"),
    ("strings", r"\b(string|substring|palindrome|character|prefix|suffix|lexicograph)\b"),
    ("geometry", r"\b(point|line segment|circle|polygon|convex.hull|intersection|collinear|triangle|distance)\b"),
    ("dsu", r"\b(union.find|disjoint.set|connected.component|merge.set|dsu)\b"),
    ("shortest paths", r"\b(shortest.path|dijkstra|bellman.ford|floyd|distance.*graph)\b"),
    ("dfs and similar", r"\b(depth.first|dfs|backtrack|recursive.*graph|flood.fill)\b"),
    ("bitmasks", r"\b(bitmask|subset|bitwise|AND.*OR|XOR.*bit|bit manipulation)\b"),
    ("two pointers", r"\b(two.pointer|sliding.window|left.*right.*pointer)\b"),
    ("hashing", r"\b(hash.map|hash.set|unordered|rolling.hash|polynomial.hash)\b"),
    ("data structures", r"\b(segment.tree|fenwick|BIT|heap|priority.queue|trie|stack|deque|sparse.table)\b"),
    ("flows", r"\b(max.flow|min.cut|network.flow|matching.*bipartite)\b"),
    ("games", r"\b(game.theory|nim|grundy|sprague|mex|optimal.*play|winning.*position)\b"),
    ("constructive algorithms", r"\b(construct|build.*answer|explicitly construct|exist.*solution)\b"),
    ("probabilities", r"\b(probability|expected.value|random|chance|likelihood)\b"),
    ("interactive", r"\b(interactive|ask.*query|queries.*answer|hidden.*number)\b"),
    ("divide and conquer", r"\b(divide.and.conquer|merge.sort|split.*half|recursive.*halve)\b"),
    ("fft", r"\b(fast.fourier|FFT|NTT|polynomial.multiplication|convolution)\b"),
    ("brute force", r"\b(all combinations|try all|brute force|exhaustive search)\b"),
    ("implementation", r"\b(simulate|simulation|grid|matrix|implement|instructions)\b"),
]


def auto_detect_tags(text: str) -> list:
    """Automatically detect CP tags from text."""
    if not text:
        return []
    # Normalize first so detection works on both raw and rendered forms
    norm = normalize_cf_text(text)
    tags = []
    for tag_name, pat in TAG_DETECTION_RULES:
        if re.search(pat, norm, re.IGNORECASE):
            tags.append(tag_name)
    return tags


# ---------------------------------------------------------------------------
# Constraint value parsing
# ---------------------------------------------------------------------------

def parse_constraint_value(raw: str) -> Optional[float]:
    """
    Parse a constraint expression to a float value.
    Handles all copy formats after normalization:
      "2*10^5"   → 200000.0
      "200000"   → 200000.0
      "1e5"      → 100000.0
      "100000"   → 100000.0
    """
    if not raw:
        return None
    raw = (
        raw.strip()
        .replace(" ", "")
        .replace(",", "")
    )
    # 10^N → 1eN
    raw = re.sub(r'10\^(\d+)', lambda m: f'1e{m.group(1)}', raw)
    # A*1eN → A*10^N evaluated
    raw = re.sub(r'(\d+)\*1e(\d+)', lambda m: str(int(m.group(1)) * 10**int(m.group(2))), raw)
    try:
        val = float(eval(raw, {"__builtins__": {}}))
        return val
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None


def extract_constraints(text: str) -> dict:
    """
    Parse constraint values from (already-normalized) problem text.
    Returns log10 of each constraint value.
    """
    features = {
        "constraint_n": 0.0,
        "constraint_m": 0.0,
        "constraint_k": 0.0,
        "constraint_q": 0.0,
        "constraint_t": 0.0,
    }

    # Normalize text before extraction (idempotent if already normalized)
    clean = normalize_cf_text(text)

    for var, patterns in CONSTRAINT_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, clean, re.IGNORECASE | re.MULTILINE)
            if m:
                val = parse_constraint_value(m.group(1))
                if val is not None and 1 <= val <= 1e18:
                    features[f"constraint_{var}"] = math.log10(max(val, 1))
                    break

    return features


# ---------------------------------------------------------------------------
# Difficulty signals
# ---------------------------------------------------------------------------

def extract_difficulty_signals(text: str) -> dict:
    """Extract difficulty-correlated keyword signals from problem text."""
    features = {}
    for tier, patterns in DIFFICULTY_KEYWORDS.items():
        hit = any(re.search(p, text, re.IGNORECASE) for p in patterns)
        features[f"diff_{tier}"] = int(hit)

    adv_count = 0
    for pat in DIFFICULTY_KEYWORDS["expert_algo"]:
        adv_count += len(re.findall(pat, text, re.IGNORECASE))
    features["advanced_algo_count"] = min(adv_count, 10)

    hard_ds_count = 0
    for pat in DIFFICULTY_KEYWORDS["hard_ds"]:
        hard_ds_count += len(re.findall(pat, text, re.IGNORECASE))
    features["hard_ds_count"] = min(hard_ds_count, 10)

    return features


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def extract_structural(record: dict) -> dict:
    """Extract structural / readability features from a problem record."""
    statement = normalize_cf_text(record.get("statement", "") or "")
    input_spec = normalize_cf_text(record.get("input_spec", "") or "")
    output_spec = normalize_cf_text(record.get("output_spec", "") or "")
    note = normalize_cf_text(record.get("note", "") or "")
    full_text = f"{statement} {input_spec} {output_spec} {note}"

    features = {}
    features["text_len"] = len(full_text)
    features["word_count"] = len(full_text.split())
    features["sentence_count"] = len(re.split(r"[.!?]", full_text))

    # Formula density (count $ or ^ or \\ as formula indicators)
    formula_matches = re.findall(r"\$[^$]+\$|\^|\bO\(", full_text)
    features["formula_count"] = len(formula_matches)
    features["formula_density"] = len(formula_matches) / max(1, features["word_count"])

    # Number of examples
    features["example_count"] = len(re.findall(
        r"(?:example|sample)\s*\d*\s*input", full_text, re.IGNORECASE
    ))

    # Paragraph count
    features["paragraph_count"] = len([p for p in full_text.split("\n\n") if p.strip()])

    # Multiple test cases
    features["has_multiple_testcases"] = int(bool(re.search(
        r"\bT\b.*test\s+case|\bT\b.*queries|first line.*test case", full_text, re.IGNORECASE
    )))

    # Modular arithmetic
    features["has_modulo"] = int(bool(re.search(
        r"mod(?:ulo)?\s+(?:\d|1e9|\$|10\^)", full_text, re.IGNORECASE
    )))
    features["has_mod_10_9"] = int(bool(re.search(
        r"10\^9\s*\+\s*7|1e9\s*\+\s*7|1000000007", full_text
    )))

    # Graph / Tree keywords
    features["has_graph_keywords"] = int(bool(re.search(
        r"\b(?:graph|node|edge|vertex|vertices|path|cycle|tree|connected|component)\b",
        full_text, re.IGNORECASE
    )))
    features["has_tree_keywords"] = int(bool(re.search(
        r"\b(?:rooted tree|binary tree|parent|child|ancestor|leaf|subtree)\b",
        full_text, re.IGNORECASE
    )))

    # Time / memory limits
    tl_str = record.get("time_limit", "") or ""
    tl_match = re.search(r"(\d+(?:\.\d+)?)", tl_str)
    features["time_limit_sec"] = float(tl_match.group(1)) if tl_match else 2.0

    ml_str = record.get("memory_limit", "") or ""
    ml_match = re.search(r"(\d+)", ml_str)
    features["memory_limit_mb"] = float(ml_match.group(1)) if ml_match else 256.0

    # ── Additional difficulty signals ─────────────────────────────────────
    constraint_vars = set(re.findall(r"\b([nmkqtxyz])\b", full_text, re.IGNORECASE))
    features["constraint_var_count"] = len(constraint_vars & set('nmkqtxyz'))

    diff_signals = extract_difficulty_signals(full_text)
    features.update(diff_signals)

    features["condition_count"] = len(re.findall(
        r"\b(?:if|when|such that|where|satisf|condition)\b", full_text, re.IGNORECASE
    ))

    features["has_count_query"] = int(bool(re.search(
        r"\b(count|number of|how many)\b", full_text, re.IGNORECASE
    )))
    features["has_optimisation"] = int(bool(re.search(
        r"\b(minimum|maximum|minimize|maximize|optimal|shortest|longest)\b", full_text, re.IGNORECASE
    )))
    features["has_existence"] = int(bool(re.search(
        r"\b(exist|possible|impossible|determine if|check whether|can you|is it possible)\b",
        full_text, re.IGNORECASE
    )))
    features["has_construct"] = int(bool(re.search(
        r"\b(construct|build|output.*sequence|print.*permutation|find.*array)\b",
        full_text, re.IGNORECASE
    )))

    features["mentions_probability"] = int(bool(re.search(
        r"\b(probability|expected|random|distribution)\b", full_text, re.IGNORECASE
    )))
    features["mentions_geometry"] = int(bool(re.search(
        r"\b(point|segment|polygon|circle|convex hull|area|perimeter)\b", full_text, re.IGNORECASE
    )))
    features["mentions_string_hard"] = int(bool(re.search(
        r"\b(suffix|palindrome|lexicograph|substring.*count|occurrence)\b", full_text, re.IGNORECASE
    )))
    features["mentions_flow"] = int(bool(re.search(
        r"\b(flow|matching|bipartite|assignment)\b", full_text, re.IGNORECASE
    )))
    features["mentions_number_theory"] = int(bool(re.search(
        r"\b(prime|coprime|gcd|lcm|euler|totient|sieve|modular inverse|divisor)\b",
        full_text, re.IGNORECASE
    )))
    features["mentions_game_theory"] = int(bool(re.search(
        r"\b(game|nim|grundy|optimal play|winning|losing)\b", full_text, re.IGNORECASE
    )))

    features["output_is_single"] = int(bool(re.search(
        r"print.*(?:one|single|the answer|a number|integer|yes|no)\b", full_text, re.IGNORECASE
    )))
    features["output_is_array"] = int(bool(re.search(
        r"print.*(?:array|sequence|permutation|list|n integer)", full_text, re.IGNORECASE
    )))

    return features


# ---------------------------------------------------------------------------
# Solution features
# ---------------------------------------------------------------------------

def extract_solution_features(solution_code: str) -> dict:
    """Extract algorithmic features from a C++ solution."""
    features = {}
    if not solution_code:
        for cat in SOLUTION_KEYWORDS:
            features[f"sol_{cat}"] = 0
        for comp in COMPLEXITY_KEYWORDS:
            safe = comp.replace("(", "").replace(")", "").replace("^", "").replace("*", "")
            features[f"complexity_{safe}"] = 0
        features["sol_lines"] = 0
        features["sol_include_count"] = 0
        return features

    code = solution_code
    for cat, patterns in SOLUTION_KEYWORDS.items():
        match = any(re.search(pat, code, re.IGNORECASE) for pat in patterns)
        features[f"sol_{cat}"] = int(match)

    for comp, patterns in COMPLEXITY_KEYWORDS.items():
        match = any(re.search(pat, code) for pat in patterns)
        safe = comp.replace("(", "").replace(")", "").replace("^", "").replace("*", "")
        features[f"complexity_{safe}"] = int(match)

    features["sol_lines"] = code.count("\n")
    features["sol_include_count"] = len(re.findall(r"#include", code))

    return features


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Fits on training data and transforms any record into a feature vector.
    All text is normalized through normalize_cf_text() before feature extraction,
    making the pipeline robust to all Codeforces copy-paste formats.
    """

    def __init__(
        self,
        tfidf_max_features: int = 15000,
        use_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.tfidf_max_features = tfidf_max_features
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self._tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            analyzer="word",
        )
        self._mlb = MultiLabelBinarizer(classes=ALL_TAGS)
        self._mlb.fit([[]])  # pre-fit with known classes
        self._embedder = None
        self._structured_cols: list = []
        self._is_fitted = False

    def _load_embedder(self):
        if self._embedder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"Loading sentence-transformer: {self.embedding_model}")
                self._embedder = SentenceTransformer(self.embedding_model)
            except ImportError:
                log.warning("sentence-transformers not installed; skipping embeddings")
                self.use_embeddings = False

    def _build_text(self, record: dict) -> str:
        """Build normalized text from a problem record."""
        parts = [
            record.get("name", "") or "",
            record.get("statement", "") or "",
            record.get("input_spec", "") or "",
            record.get("output_spec", "") or "",
            record.get("note", "") or "",
        ]
        raw = " ".join(p for p in parts if p).strip()
        return normalize_cf_text(raw)

    def fit(self, records: list) -> "FeaturePipeline":
        """Fit TF-IDF on training records (normalized text)."""
        texts = [self._build_text(r) for r in records]
        self._tfidf.fit(texts)
        log.info(f"  TF-IDF fitted: {len(self._tfidf.vocabulary_):,} features")

        sample_structured = self._extract_structured_dict(records[0], solution_code=None)
        self._structured_cols = list(sample_structured.keys())
        log.info(f"  Structured features: {len(self._structured_cols)}")

        self._is_fitted = True
        return self

    def _extract_structured_dict(self, record: dict, solution_code: Optional[str]) -> dict:
        full_text = " ".join([
            normalize_cf_text(record.get("statement", "") or ""),
            normalize_cf_text(record.get("input_spec", "") or ""),
            normalize_cf_text(record.get("output_spec", "") or ""),
            normalize_cf_text(record.get("note", "") or ""),
        ])
        d = {}
        d.update(extract_constraints(full_text))
        d.update(extract_structural(record))
        d.update(extract_solution_features(solution_code or ""))

        # Derived constraint features
        cn = d.get("constraint_n", 0.0)
        cq = d.get("constraint_q", 0.0)
        ct = d.get("constraint_t", 0.0)
        d["constraint_nq_sum"] = cn + cq        # log10(n*q) — large = harder queries
        d["constraint_nt_sum"] = cn + ct        # log10(n*t) — large = harder with t test cases
        d["constraint_n_small"]  = float(cn > 0 and cn <= 3.0)   # n <= 1000
        d["constraint_n_medium"] = float(cn > 3.0 and cn <= 5.0) # 1000 < n <= 1e5
        d["constraint_n_large"]  = float(cn > 5.0 and cn <= 6.0) # 1e5 < n <= 1e6
        d["constraint_n_huge"]   = float(cn > 6.0)               # n > 1e6
        d["constraint_n_squared"] = float((cn * 2) > 9.0)        # n^2 > 1e9
        return d

    def transform_single(self, record: dict, solution_code: Optional[str] = None) -> dict:
        """Transform a single record to feature dict (for inference)."""
        text = self._build_text(record)

        tfidf_vec = self._tfidf.transform([text])

        # Tags — auto-detect from normalized text if empty
        tags = record.get("tags") or []
        if not tags:
            tags = auto_detect_tags(text)
        tag_vec = self._mlb.transform([tags])

        structured = self._extract_structured_dict(record, solution_code)

        emb_vec = None
        if self.use_embeddings:
            self._load_embedder()
            if self._embedder:
                emb_vec = self._embedder.encode([text[:512]], normalize_embeddings=True)

        return {
            "tfidf": tfidf_vec,
            "tags": tag_vec,
            "structured": structured,
            "embedding": emb_vec,
            "detected_tags": tags,
        }

    def transform_batch(
        self,
        records: list,
        solution_codes: Optional[list] = None,
        batch_size: int = 64,
    ) -> dict:
        """Transform a list of records; returns dict of arrays."""
        if solution_codes is None:
            solution_codes = [None] * len(records)

        texts = [self._build_text(r) for r in records]
        tags_list = []
        for r, t in zip(records, texts):
            tg = r.get("tags") or []
            if not tg:
                tg = auto_detect_tags(t)
            tags_list.append(tg)

        tfidf_mat = self._tfidf.transform(texts)
        tag_mat = self._mlb.transform(tags_list)

        structured_rows = []
        for rec, sol in zip(records, solution_codes):
            d = self._extract_structured_dict(rec, sol)
            row = [d.get(col, 0.0) for col in self._structured_cols]
            structured_rows.append(row)
        struct_mat = np.array(structured_rows, dtype=np.float32)
        struct_mat = np.nan_to_num(struct_mat, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            "tfidf": tfidf_mat,
            "tags": tag_mat,
            "structured": struct_mat,
            "structured_cols": self._structured_cols,
            "embedding": None,
        }

        if self.use_embeddings:
            self._load_embedder()
            if self._embedder:
                log.info("Computing sentence embeddings …")
                from tqdm import tqdm
                embeddings = []
                for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
                    batch = texts[i: i + batch_size]
                    batch = [t[:512] for t in batch]
                    emb = self._embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                    embeddings.append(emb)
                result["embedding"] = np.vstack(embeddings).astype(np.float32)

        return result

    def build_feature_matrix(self, feature_dict: dict) -> sp.csr_matrix:
        """Combine all feature groups into a single sparse matrix."""
        parts = [feature_dict["tfidf"], feature_dict["tags"].astype(np.float32)]

        struct = feature_dict["structured"]
        if isinstance(struct, dict):
            struct = np.array(
                [[struct.get(col, 0.0) for col in self._structured_cols]],
                dtype=np.float32,
            )
        struct = np.nan_to_num(struct, nan=0.0, posinf=0.0, neginf=0.0)

        if feature_dict.get("embedding") is not None:
            emb_sparse = sp.csr_matrix(feature_dict["embedding"].astype(np.float32))
            parts.append(emb_sparse)

        struct_sparse = sp.csr_matrix(struct)
        parts.append(struct_sparse)

        return sp.hstack(parts, format="csr")

    def save(self, path: str):
        import joblib
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._tfidf, p / "tfidf.pkl")
        joblib.dump(self._mlb, p / "mlb.pkl")
        joblib.dump(self._structured_cols, p / "structured_cols.pkl")
        joblib.dump({
            "tfidf_max_features": self.tfidf_max_features,
            "use_embeddings": self.use_embeddings,
            "embedding_model": self.embedding_model,
        }, p / "config.pkl")
        log.info(f"Pipeline saved to {p}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        import joblib
        p = Path(path)
        config = joblib.load(p / "config.pkl")
        pipeline = cls(**config)
        pipeline._tfidf = joblib.load(p / "tfidf.pkl")
        pipeline._mlb = joblib.load(p / "mlb.pkl")
        pipeline._structured_cols = joblib.load(p / "structured_cols.pkl")
        pipeline._is_fitted = True
        return pipeline

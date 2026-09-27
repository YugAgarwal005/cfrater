# CFRater

**Estimate the Codeforces difficulty rating of any competitive programming problem using machine learning.**

An ML ensemble trained on real Codeforces problem data that predicts difficulty ratings from problem text, constraints, tags, and solution code. All predictions are estimates — not official Codeforces ratings.

Live at: **[cfrater.onrender.com](https://cfrater.onrender.com)**

---

## How it works

1. Paste a problem statement, constraints, and tags
2. Optionally paste a reference C++ solution
3. The ensemble of XGBoost, LightGBM, Random Forest, and Ridge models predicts the difficulty rating
4. A stacking meta-learner and isotonic calibrator refine the final output

Supports all Codeforces copy-paste formats — raw LaTeX (`$n \leq 2 \cdot 10^5$`), browser-rendered text (`n <= 2*10^5`), and the superscript-loss bug (`105` instead of `10^5`).

---

## Local setup

```bash
git clone https://github.com/YugAgarwal005/cfrater.git
cd cfrater
pip install -r requirements.txt
python src/api/app.py
# open http://localhost:5000
```

## Retrain the model

```bash
# Collect fresh data from Codeforces API
python src/data_collection/scraper.py --output data/raw/problems.jsonl

# Train with 4x data augmentation
python src/models/train.py --data data/processed/codeforces_full.jsonl --output models/ --augment 4
```

---

## ML approach

### Feature groups
| Group | Details |
|---|---|
| TF-IDF text | 18,000 features from normalized problem text |
| Tags | Multi-hot encoding of 39 Codeforces topic tags |
| Constraints | Parsed `n`, `m`, `k`, `q`, `t` bounds (log scale) |
| Structural | Text length, formula density, condition count, keyword signals |
| Solution | Detected algorithms and data structures from C++ code |

### Models
| Model | Role |
|---|---|
| Ridge Regression | Fast baseline, high weight in meta-learner |
| XGBoost Regressor | Best on structured features |
| LightGBM Regressor | Best single model (MAE 366) |
| Random Forest | Regressor + Classifier |
| Stacking Meta (Ridge) | Combines base model predictions |
| Isotonic Calibrator | Corrects range compression bias |

**Final calibrated ensemble: MAE ~341, R2 0.57**

### Data augmentation
Every training problem is augmented into 4 variants to make the model robust to all copy-paste formats:
1. Original (raw Codeforces text with LaTeX/markdown)
2. Rendered (LaTeX normalized to plain text)
3. Mangled (simulated superscript-loss bug)
4. Plain-text (markdown stripped, whitespace cleaned)

---

## Disclaimer

This tool is not affiliated with Codeforces. Ratings shown are ML estimates based on historical data and are not official Codeforces difficulty ratings.

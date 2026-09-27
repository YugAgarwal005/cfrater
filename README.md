# 🏆 Codeforces Rating Predictor

> An ML-based system that predicts the estimated difficulty rating of competitive programming problems using real Codeforces data.

**⚠️ Disclaimer:** This system provides *estimated difficulty ratings* based on historical Codeforces data. These are **not official Codeforces ratings**.

---

## 📁 Project Structure

```
rating-predictor/
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Scraped from Codeforces API
│   └── processed/            # Feature-engineered data
├── src/
│   ├── data_collection/      # Codeforces API scrapers
│   ├── feature_engineering/  # Feature extraction pipeline
│   ├── models/               # ML model training & evaluation
│   └── api/                  # Flask/FastAPI prediction server
├── web/                      # Beautiful web UI
├── notebooks/                # EDA & experimentation
├── models/                   # Saved trained models
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect data from Codeforces
```bash
python src/data_collection/scraper.py --output data/raw/problems.jsonl
```

### 3. Build features & train models
```bash
python src/models/train.py --data data/raw/problems.jsonl --output models/
```

### 4. Launch the web UI
```bash
python src/api/app.py
# then open http://localhost:5000
```

---

## 🧠 ML Approach

### Features Used
- **Problem text**: TF-IDF + sentence-transformer embeddings
- **Tags**: Multi-hot encoded topic tags
- **Constraints**: Parsed `n`, `m`, `k` bounds, time/memory limits
- **Structural**: Problem length, number of examples, formula density
- **Solution analysis**: Detected algorithms, data structures, complexity keywords
- **Contest metadata**: Problem index (A/B/C/D/E), contest date

### Models Trained
| Model | Type | Notes |
|-------|------|-------|
| Random Forest | Classification | Strong baseline |
| Gradient Boosting (XGBoost) | Classification + Regression | Best structured features |
| Ridge Regression | Regression | Interpretable |
| NN with embeddings | Regression | Best with text features |

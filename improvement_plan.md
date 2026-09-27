# CF Rating Predictor — Improvement Plan

---

## 🔴 Why It Always Returns 1500

**Short answer: No data has been collected and no models have been trained yet.**

The system is running in **demo mode**, which returns a hardcoded example response (`predicted_rating: 1500`) regardless of what you paste. This is by design — demo mode exists so the UI works before training is done.

**Status right now:**
| Component | Status |
|-----------|--------|
| Data collected (`data/raw/problems.jsonl`) | ❌ Empty — never run |
| Models trained (`models/*.pkl`) | ❌ None exist |
| Server mode | 🟡 **Demo mode** (always returns 1500) |

### Fix: Run data collection + training

```powershell
$py = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe"

# Step 1 — Collect ~5000 problems (no HTML = fast, ~5 min)
& $py src\data_collection\scraper.py --output data\raw\problems.jsonl --no-statements --max-problems 5000

# Step 2 — Train (fast, no embeddings)
& $py src\models\train.py --data data\raw\problems.jsonl --output models\ --no-embeddings

# Done — restart the server, it will use real predictions
```

> Collecting with `--no-statements` gets all tags + ratings via the API in minutes. Full HTML statements take 30–60 min but improve accuracy by ~10–15%.

---

## 📋 Planned Improvements

### 1. 🏷️ Auto Tag Generation from Problem Text
**What:** The system should analyze the pasted problem statement and suggest probable Codeforces tags (dp, greedy, graphs, etc.) using the trained model or a rule-based classifier.

**How:**
- Train a **multi-label classifier** (tags as targets) alongside the rating model
- Use TF-IDF + keyword patterns to suggest tags in real-time as the user types
- Show suggestions as clickable chips the user can confirm/reject
- Also use the `autoDetectTags()` function already in `app.js` (currently only fires on paste), make it work live

**Files:** `src/models/train.py`, `src/models/predictor.py`, `web/app.js`

---

### 2. 📝 Replace Constraint Fields with a Single Text Area
**What:** Remove the 4-box constraint grid (n, q, time limit, memory) and replace it with a single free-text `Constraints` field where you paste the constraint block directly from the problem.

**Before:**
```
[n input] [q input] [time limit input] [memory limit input]
```

**After:**
```
Constraints (paste directly from problem):
┌─────────────────────────────────────────────┐
│ 1 ≤ n ≤ 2·10^5                              │
│ 1 ≤ q ≤ 10^5                                │
│ Time limit: 2 seconds                        │
│ Memory limit: 256 megabytes                  │
└─────────────────────────────────────────────┘
```

The backend already parses constraints from raw text — so no ML changes needed, just a UI change in `web/index.html` + `web/style.css`.

**Files:** `web/index.html`, `web/style.css`, `web/app.js`

---

### 3. ❌ Remove Problem Position Selector (A/B/C/D/E/F)
**What:** Remove the A/B/C/D/E/F button row entirely. It adds complexity to the UI and the model already infers difficulty from the text and tags anyway.

**Why it's fine to remove:** Contest position is a weak signal (the feature just encodes a number 1–6). Tags and text features dominate.

**Files:** `web/index.html`, `web/style.css`, `web/app.js`

---

### 4. 🤖 Make Demo Mode Smarter (Pre-training)
**What:** Instead of always returning exactly 1500, make demo mode do a rough rule-based prediction using:
- Detected tags → approximate bucket
- Constraint size (n ≤ 10^3 = likely easy, n ≤ 10^5 = medium, n ≤ 10^18 = likely hard)
- Solution complexity keywords

This way users get *something useful* even before training.

**Files:** `src/api/app.py`

---

### 5. 📊 Show Training Progress in the UI
**What:** Add a `/api/training-status` endpoint and a small status bar in the UI showing:
- How many problems have been collected
- Whether models are trained
- Training accuracy from last run

**Files:** `src/api/app.py`, `web/index.html`, `web/app.js`

---

### 6. 🔁 Background Data Collection Endpoint
**What:** Allow triggering data collection from the web UI with a "Collect Data" button that shows live progress without needing to use the terminal.

**Files:** `src/api/app.py`, `web/index.html`, `web/app.js`

---

### 7. 🎯 Improve Model Accuracy
**Current gap:** Without HTML statements, tag-only models have ~±300 rating MAE. With statements + embeddings it drops to ~±180.

**Planned improvements:**
- Add a **stacking meta-learner** (trains on top of base model predictions)
- Use **rating bucket weighting** (harder to predict extreme ratings — weight them higher)
- Experiment with `all-mpnet-base-v2` embedding model (larger, more accurate)
- Add **cross-validation** to pick better hyperparameters

---

## 🗓️ Execution Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 🔴 **Now** | Run data collection + training | Run 2 commands | Fixes 1500-always bug |
| 🟠 **Next** | Remove position selector | 10 min | UX |
| 🟠 **Next** | Replace constraint boxes with text area | 20 min | UX |
| 🟡 **After** | Auto tag generation (rule-based) | 30 min | UX + accuracy |
| 🟡 **After** | Smarter demo mode | 20 min | UX |
| 🟢 **Later** | Training status in UI | 1 hr | UX |
| 🟢 **Later** | Background collection trigger | 1–2 hr | UX |
| 🔵 **Later** | Multi-label tag classifier | 2–3 hr | Accuracy |
| 🔵 **Later** | Stacking meta-learner | 2 hr | Accuracy |

---

> Click **Proceed** to implement all the UI changes (Priority 🟠 + 🟡) immediately, and also start data collection in the background.

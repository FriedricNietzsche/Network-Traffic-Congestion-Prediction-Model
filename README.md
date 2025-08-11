# Network Traffic Congestion Prediction & Monitoring Dashboard

End‑to‑end machine learning and monitoring system for predicting network traffic congestion. The application covers synthetic data generation, feature engineering, imbalance handling, model training and tuning, explainability (local + global SHAP), version management (save, list, rollback, retention), asynchronous training with server‑sent events (SSE) progress streaming, probability threshold control, and an interactive Flask dashboard. A lightweight automated test suite is included.

## Key Features
* Synthetic data generator (deterministic seed in code for reproducibility).
* Feature engineering: IP to integer, cyclical hour (sin/cos), temporal aggregates, weekend flag, protocol one‑hot encoding.
* Configurable class imbalance handling (SMOTE or random oversampling) and class_weight support.
* Hyperparameter randomized search with time series cross‑validation (per config) for model selection.
* Supported models: RandomForest (default) and optional XGBoost (if installed).
* Asynchronous training endpoint with SSE progress events and per‑phase timing (data, feature engineering, training, scoring, SHAP, version save).
* Synchronous pipeline script (`scripts/run_pipeline.py`) for full offline regeneration.
* Model versioning: timestamped artifacts (model / scaler / metrics / SHAP top) with retention pruning and rollback endpoint.
* Local SHAP explanations: probability, decision, expected value, feature contributions, original feature values, optional cyclical pair collapse, cumulative percentages.
* Global mean |SHAP| importance pre‑computation and caching.
* Interactive dashboard: metrics cards, trends (F1 / Precision / Recall), confusion matrix counts trend, SHAP visualizations, threshold slider, version table (download + rollback), toast notifications.
* Prediction page with curated presets and randomization aligned to the training distribution, plus adjustable inference threshold.
* Defensive inference alignment: ensures feature order consistency; fills or drops columns as required.
* Dark mode toggle, auto refresh, artifact downloads (model, scaler, metrics, versioned models).
* Logging to file and console; extensible YAML configuration.
* Automated pytest suite (data generation, feature engineering, prediction, API, explainability, presets) with warning suppression.

## Architecture Overview
* Data: Synthetic generation (`src/data/generate_data.py`).
* Features: Transformation and scaling (`src/features/feature_engineering.py`).
* Modeling: Training logic with CV and imbalance handling (`src/models/train_model.py`).
* Inference: Prediction + feature alignment (`src/models/predict.py`).
* Explainability: SHAP local/global endpoints in Flask.
* Orchestration: Flask app (`app.py`) plus async training background thread + SSE.
* Versioning: Timestamped artifact persistence + retention pruning + rollback.
* Visualization: Static matplotlib plots and dynamic Chart.js dashboard views.
* Testing: Pytest suite in `tests/`.

## Project Structure
```
.
├── app.py                         # Flask application (UI, APIs, async training, rollback)
├── requirements.txt
├── README.md
├── config/
│   └── config.yaml                # Central configuration (model, imbalance, api, versioning, paths)
├── scripts/
│   └── run_pipeline.py            # End-to-end synchronous pipeline
├── src/
│   ├── data/generate_data.py
│   ├── features/feature_engineering.py
│   ├── models/train_model.py
│   ├── models/predict.py
│   ├── utils/logger.py
│   └── visualization/visualize.py
├── models/                        # Current model, scaler, metrics, history, shap_global.json, versions/
├── models/versions/               # model_<ts>.pkl, scaler_<ts>.pkl, metrics_<ts>.json
├── visualization/                 # Generated PNG plots (EDA & evaluation)
├── templates/                     # Jinja2 templates (dashboard, predict, base, index)
├── static/                        # CSS and static assets
├── tests/                         # Automated pytest suite
├── notebooks/                     # Exploratory analysis notebooks
└── logs/                          # Application logs
```

## Configuration (excerpt)
```yaml
data:
    raw_path: data/traffic_data.csv
    test_size: 0.2
    random_state: 42
model:
    primary: random_forest
    cv_splits: 5
    search_iterations: 15
    random_forest:
        param_grid:
            n_estimators: [100, 200, 400]
            max_depth: [null, 10, 20, 40]
    imbalance:
        enable: true
        method: smote
api:
    default_threshold: 0.5
versioning:
    enable: true
    directory: models/versions
    keep_last: 10
paths:
    model_path: models/congestion_model.pkl
    scaler_path: models/scaler.pkl
    features_path: models/X.pkl
```

## API Reference (selected)
| Route | Method | Description | Notes |
|-------|--------|-------------|-------|
| `/api/predict` | POST | Predicts congestion | Query param `threshold` overrides default |
| `/api/metrics` | GET | Latest metrics snapshot | precision, recall, f1_score, auc_roc, counts |
| `/api/metrics/history` | GET | Recent training runs | Deduped, capped length |
| `/api/train` | POST | Synchronous training | Optional `?generate_data=true` |
| `/api/train/async` | POST | Start background training | Returns started status |
| `/api/train/events` | GET | SSE progress stream | EventSource in dashboard |
| `/api/explain` | POST | Local SHAP explanation | Params: `n`, `collapse_pairs`, `threshold` |
| `/api/explain/global` | GET | Global SHAP importance | 404 if unavailable |
| `/api/versions/detail` | GET | Version list (metrics enriched) | Includes shap_top |
| `/api/versions/rollback/<timestamp>` | POST | Roll back active model | Timestamp YYYYMMDD_HHMMSS |
| `/download/model` | GET | Current model |  |
| `/download/scaler` | GET | Current scaler |  |
| `/download/metrics` | GET | Latest metrics |  |
| `/download/version/<file>` | GET | Specific versioned model |  |

Example invocation:
```bash
curl -X POST "http://localhost:5000/api/predict?threshold=0.55" \
    -H "Content-Type: application/json" \
    -d '{"packet_size":600,"bytes_sent":2400,"source_ip":"192.168.1.10","dest_ip":"10.0.0.50","protocol":"TCP","timestamp_seconds":1700000000,"hour":12}'
```

## Explainability
* Global SHAP importance computed after training for tree‑based models (cached at `models/shap_global.json`).
* Local explanation returns probability, applied threshold, decision, expected value, feature contributions, original feature values, and cumulative percentages.
* Cyclical features (hour_sin, hour_cos) can be collapsed logically in the UI.

## Quick Start (Windows PowerShell)
```powershell
# 1. (Optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) full pipeline (regenerates data + trains + plots + global SHAP)
python scripts/run_pipeline.py --generate-data

# 4. Run the dashboard/API
python app.py
# Visit http://127.0.0.1:5000
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --generate-data
python app.py
```

## Training and Retraining
Two primary paths:
1. Scripted pipeline: complete regeneration (data → feature engineering → train → metrics → plots → SHAP).
2. Asynchronous web retrain: `/api/train/async` with SSE stream (`/api/train/events`) for live progress. Per‑phase durations and total time stored in version metrics.

## Model Versioning and Rollback
* Each async run (if enabled) saves `model_<timestamp>.pkl`, `scaler_<timestamp>.pkl`, `metrics_<timestamp>.json` (with phase durations and SHAP top features).
* Retention pruning preserves the most recent N versions (`keep_last`).
* Rollback endpoint restores selected version into active artifacts and appends a rollback event to metrics history.

## Probability Threshold Control
Threshold can be specified via dashboard slider or query parameter. Decision boundary adjustments do not require retraining and are reflected in explain responses.

## Dashboard Components
* Metrics cards: precision, recall, F1, AUC, samples, features.
* Trend lines: F1 / Precision / Recall over recent runs.
* Confusion matrix counts trend.
* Local SHAP bar chart (recent explanation sample).
* Global SHAP bar chart.
* Feature contribution table: raw values, signed SHAP, percent, cumulative percent, contribution bar, CSV export.
* Version table: timestamp, key metrics, SHAP top features, download & rollback actions.
* Probability threshold slider, retrain controls, dark mode toggle, toast notifications.

## Testing
Run all tests:
```bash
python -m pytest -q
```
Included tests:
* Data generation (column presence, row count).
* Feature engineering (cyclical features, protocol one‑hot, target retention).
* Prediction function (probability bounds, valid class output).
* API endpoints (predict, metrics).
* Local SHAP explanation (xfail if unsupported environment).
* Preset prediction scenarios (probability sanity checks).

## Logging
Central logger (`src/utils/logger.py`) outputs to console and log file under `logs/` (configured in `config.yaml`).

## Primary Artifacts
| Path | Description |
|------|-------------|
| models/congestion_model.pkl | Active model |
| models/scaler.pkl | Active scaler |
| models/metrics.json | Latest metrics snapshot |
| models/metrics_history.json | Performance history + rollback events |
| models/shap_global.json | Global SHAP importance cache |
| models/versions/ | Versioned artifact sets |
| visualization/*.png | Static EDA and evaluation plots |

## Suggested .gitignore Entries
```
.venv/
__pycache__/
*.pyc
models/*.pkl
models/metrics.json
models/metrics_history.json
models/shap_global.json
logs/
visualization/
```

## Roadmap (Future Enhancements)
| Area | Planned Improvement |
|------|---------------------|
| Drift & Alerting | Surface significant drops in F1 / AUC with UI indicators |
| Security | Input validation, auth, rate limiting |
| Incremental Learning | Explore partial_fit or streaming model adaptation |
| CI/CD | GitHub Actions workflow (tests, lint, build) |
| OpenAPI | Generated specification for public API consumption |
| Data Quality | Scenario parameterization & shift simulation |

## License
Idk and idc

## Acknowledgements
Python ecosystem: scikit-learn, Flask, SHAP, XGBoost, Chart.js, and associated open source libraries. (These dependencies will be reduced to ashes in my next project)

---
Contributions and issue reports are welcome.

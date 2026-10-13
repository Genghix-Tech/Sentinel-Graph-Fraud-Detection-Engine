# SentinelGraph: AI Fraud & Anomaly Investigation Engine

[![CI](https://github.com/Genghix-Tech/Sentinel-Graph-Fraud-Detection-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Genghix-Tech/Sentinel-Graph-Fraud-Detection-Engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GraphSAGE-EE4C2C?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189FDD)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **Discover hidden fraud patterns that ordinary transaction classifiers miss.**

SentinelGraph doesn't just flag suspicious transactions. It **investigates** the relationships between accounts, devices, IPs, cards and shipping addresses, and explains *why* an event is suspicious: contributing factors, a linked-entity graph, an investigation timeline and a plain-English case narrative. Analyst decisions feed back into the next model version.

![SentinelGraph](docs/screenshots/00-cover.png)

**Stack:** Python · FastAPI · XGBoost · PyTorch Geometric (GraphSAGE) · NetworkX · SHAP · scikit-learn · PostgreSQL · SQLAlchemy · React · TypeScript · Recharts · Docker

---

## Results

Evaluated on a held-out time window (the last 20% of 32,607 transactions) at the **same alert volume**, 3% of traffic, which is what an analyst team could realistically review:

| Model | PR-AUC | Recall @ 3% | Fraud ring | Account takeover | Card testing |
|---|---:|---:|---:|---:|---:|
| Tabular XGBoost (*ordinary classifier baseline*) | 0.804 | 75.3% | 62.0% | 93.8% | 100% |
| Isolation Forest (unsupervised) | 0.904 | 91.1% | 91.0% | 68.8% | 100% |
| GraphSAGE GNN (account graph) | 0.598 | 59.5% | 91.0% | 6.2% | 4.8% |
| XGBoost + point-in-time graph features | 0.996 | 99.4% | 99.0% | 100% | 100% |
| **SentinelGraph ensemble** | **0.996** | **100%** | **100%** | **100%** | **100%** |

- **39 fraudulent transactions ($3.9k) caught that the tabular baseline missed** at the same alert budget. 38 of them belonged to fraud rings whose individual purchases look ordinary.
- At its production threshold the ensemble alerts on 2.4% of traffic with **99.4% precision and 98.1% recall**.
- The #1 SHAP driver is a graph feature (linked accounts with confirmed fraud), as is #4 (how tightly the cluster's sign-up dates are grouped).
- Each detector covers a different pattern: the GNN is strong on rings but blind to account takeover, and the anomaly detector catches card testing. The ensemble combines them.
- Results are fully reproducible: `python -m app.cli seed` with the default seed gives exactly these numbers.

> Data is **synthetic** (see [The data](#the-data)), so absolute numbers are optimistic. The *gap* between the tabular and graph models is the point: the generator deliberately makes fraud rings look normal one transaction at a time.

---

## Screenshots

| | |
|---|---|
| ![Overview](docs/screenshots/01-overview-dashboard.png) **Live fraud operations dashboard** | ![Attack detected](docs/screenshots/02-live-attack-detected.png) **Simulated fraud ring detected in real time** |
| ![Explainable alert](docs/screenshots/04-explainable-alert.png) **Explainable alert: SHAP factors + case narrative** | ![Entity graph](docs/screenshots/05-linked-entity-graph.png) **New ring linked to 3 known rings via recycled devices** |
| ![Alert queue](docs/screenshots/03-alert-queue.png) **Alert queue: behavioural vs. graph-driven** | ![Scoring API](docs/screenshots/06-realtime-scoring-api.png) **Real-time scoring API** |
| ![Evaluation](docs/screenshots/07-model-evaluation.png) **Recall by pattern & SHAP feature importance** | ![Comparison](docs/screenshots/08-model-comparison.png) **5-model benchmark & version history** |

---

## What it does

| Capability | How |
|---|---|
| **Graph-based detection** | A heterogeneous NetworkX graph (`account ↔ device / ip / card / address`) built **incrementally in time order**. Every transaction is featurised against the graph *as it existed at that moment*. |
| **Supervised + unsupervised** | XGBoost on 40 behavioural + graph features, Isolation Forest for novel anomalies, and a GraphSAGE GNN (PyTorch Geometric) for account-level relational risk. Blend weights are tuned on a validation window. |
| **Explainable alerts** | SHAP TreeExplainer contributions turned into human-readable factors ("Device is shared by 8 accounts"), a case narrative, and model-by-model signals. |
| **Investigation timeline** | Account opening, first sightings of each device/card/address, linked accounts opening and being confirmed as fraud, surrounding transactions, the alert, and analyst decisions. |
| **Linked-entity graph** | Interactive force graph of the account's cluster. Shape and colour encode entity type; confirmed-fraud and alerted accounts are highlighted. |
| **Analyst feedback loop** | Confirm/dismiss decisions (1) immediately propagate "known fraud" through the live graph, and (2) override chargeback labels in the next retrain, which produces a new, versioned model. |
| **Live dashboard** | KPIs, alerts by severity, suspected fraud rings, live transaction feed, score distribution. Inject a fraud ring, account takeover or card-testing attack and watch it get caught in real time. |
| **Real-time scoring API** | `POST /api/transactions` featurises against the live graph, scores, persists and alerts with a SHAP explanation in one call (~250 ms end to end on a laptop with SQLite). |

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
      SIM[Simulator / clients] -->|POST /api/transactions| ENG
    end
    subgraph Engine["Real-time engine (FastAPI)"]
      ENG[FeatureState<br/>point-in-time features] --> G[(NetworkX<br/>entity graph)]
      G --> ENG
      ENG --> XGB[XGBoost]
      ENG --> IF[Isolation Forest]
      GNN[GraphSAGE GNN<br/>periodic snapshots] --> ENS
      XGB --> ENS{Weighted<br/>ensemble}
      IF --> ENS
      ENS -->|above threshold| EXP[SHAP + narrative]
    end
    EXP --> DB[(PostgreSQL)]
    ENS --> DB
    DB --> UI[React dashboard]
    UI -->|analyst feedback| DB
    DB -->|labels + feedback| TRAIN[Training pipeline] -->|new version| ENG
```

**Design decisions worth noting**

- **No leakage.** Graph features are computed incrementally, so a transaction never sees links that formed after it. Fraud labels arrive with a 10-day **chargeback delay**, so "linked to confirmed fraud" only counts labels that were actually known at that time. The train/validation/test split is by time.
- **One feature code path.** The same `FeatureState` class replays history for training and scores live traffic, so there's no training/serving skew.
- **GNN as a batch job.** GraphSAGE runs over graph snapshots (every 6 hours in evaluation; periodically in production). Accounts newer than the last snapshot borrow their cluster's GNN risk, and links that just formed are covered by the real-time graph features.
- **Graceful degradation.** If PyTorch Geometric isn't installed, an SGC-style graph model (logistic regression on `[X, ÂX, Â²X]`) takes its place. SHAP falls back to XGBoost's native TreeSHAP if library versions disagree.
- **Equal-budget evaluation.** Models are compared at the same alert volume, not just by AUC, because analyst capacity is the real constraint.

---

## The data

`app/data/generator.py` builds a 90-day world of ~2,700 customers and ~33k transactions. Legitimate behaviour is deliberately messy so that no single signal gives fraud away:

- **Benign sharing:** families share a tablet, wifi and address; offices put 15–40 strangers behind one NAT IP; phones get resold.
- **Hard negatives:** student housing, where new accounts opened within days of each other share an address and wifi. That looks structurally like a fraud ring.
- **Normal noise:** travel abroad, new phones, gifts shipped elsewhere, new customers joining mid-window.

Injected fraud scenarios:

| Pattern | Looks like | Why it's hard |
|---|---|---|
| **Fraud ring** (synthetic identities) | 4–10 accounts sharing devices, drop addresses and stolen cards; half are aged "sleeper" accounts; devices get recycled across rings | Individual purchases have ordinary amounts, categories and hours |
| **Account takeover** | Established account suddenly transacts from a new device in another country, at night, for 6× usual spend; goods go to mule addresses | Strong per-transaction signal, but the mule addresses link victims |
| **Card testing** | Burst of $0.50–$5 charges on many cards, then a large purchase; proxies and devices reused | Velocity and anomaly signal |

---

## Quick start

```bash
git clone https://github.com/Genghix-Tech/Sentinel-Graph-Fraud-Detection-Engine.git
cd Sentinel-Graph-Fraud-Detection-Engine
```

### Option A: Docker (PostgreSQL)

```bash
docker compose up --build
```

On first start the API seeds the database and trains model v1 (~1–2 min). Open **http://localhost:8080** for the dashboard and **http://localhost:8000/docs** for the API docs.

### Option B: local (SQLite, no setup)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -r requirements-gnn.txt                  # optional: GraphSAGE (else SGC fallback)
python -m app.cli seed                               # generate data, train v1, raise alerts
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev                                          # http://localhost:5173 (proxies /api to :8000)
```

Set `DATABASE_URL=postgresql+psycopg2://user:pass@host/db` to use PostgreSQL locally.

### Tests

```bash
cd backend && GNN_BACKEND=sgc pytest -q
```

They cover point-in-time correctness, label-delay handling, velocity/novelty features, the generator, and the API flow end to end (seed → investigate → feedback → simulate attack → real-time scoring).

---

## Demo script (2 minutes)

1. **Overview:** the hero banner shows the fraud the tabular baseline missed. Click **Fraud ring** to inject a new ring that reuses a known fraudster's device, then click **Investigate** in the toast.
2. **Investigation:** read the narrative and SHAP factors, explore the linked-entity graph (the new ring connects to a historical one), and walk the timeline.
3. Click **Confirm fraud**. That account now counts as known fraud for every linked account's live score.
4. **Models:** compare the five models, see which features drive decisions, then click **Retrain with feedback** to produce a new model version from analyst labels.
5. Toggle **Live stream** for continuous traffic with occasional attacks.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/overview` | KPIs, daily alerts, suspected rings, live feed, score histogram |
| `GET` | `/api/alerts` | Alert queue (filter by `status`, `severity`, `account_id`; sort by `risk`/`recent`) |
| `GET` | `/api/alerts/{id}` | Explanation, narrative, timeline, context |
| `GET` | `/api/alerts/{id}/graph` | Linked-entity subgraph |
| `POST` | `/api/alerts/{id}/feedback` | `{"label": "fraud" \| "legit", "notes": "..."}` |
| `POST` | `/api/transactions` | Real-time scoring of a transaction |
| `GET` | `/api/models` | Model versions, metrics, feature importance |
| `POST` | `/api/models/retrain` | Train a new version with analyst feedback (async) |
| `POST` | `/api/simulate` | `{"scenario": "ring" \| "ato" \| "card_testing" \| "legit"}` |
| `POST` | `/api/stream` | `{"enabled": true}` toggles continuous live traffic |

---

## Project layout

```
backend/
  app/
    data/generator.py        synthetic world + fraud scenarios
    graph/entity_graph.py    incremental NetworkX entity graph, clusters, investigation subgraph
    graph/gnn.py             account snapshots, GraphSAGE (PyG) + SGC fallback
    ml/features.py           point-in-time feature engine (40 features)
    ml/pipeline.py           training, ensemble tuning, equal-budget evaluation, versioning
    ml/bundle.py             model bundle: scoring, thresholds, SHAP
    ml/explain.py            factor descriptions + case narratives
    services/engine.py       real-time scoring engine
    services/investigation.py timeline builder
    services/simulator.py    live traffic + attack injection
    api.py, main.py          FastAPI
    cli.py                   seed / train
  tests/
frontend/
  src/pages/                 Overview, Alerts, Investigation, Models
  src/components/            EntityGraph (force graph), UI primitives
docker-compose.yml           Postgres + API + dashboard
```

## Limitations & next steps

- Synthetic data only. The next step is a public dataset such as IEEE-CIS or Elliptic.
- The in-memory graph suits a single node at this scale. In production it would be a graph store (Neo4j, or TigerGraph for streaming) with features pushed to a feature store.
- Retraining runs in-process. A real deployment would use a job queue and a model registry (MLflow) with shadow scoring before promotion.

## License

MIT

"""Training, evaluation and versioning.

Time-based split (train 70% / validation 10% / test 20% of the timeline). Four models:

* XGBoost on per-transaction (tabular) features only      -> the "ordinary classifier" baseline
* XGBoost on tabular + point-in-time graph features
* Isolation Forest (unsupervised anomaly detection)
* GraphSAGE GNN over the account graph

Ensemble weights are tuned on validation, the alert threshold is set from the analyst
review budget, and every model is compared on the untouched test window at the same
alert volume. Analyst feedback overrides chargeback labels and is folded into training.
"""
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import ALERT_BUDGET, LABEL_DELAY_DAYS, SEED
from app.graph.gnn import GraphRiskModel, build_snapshot
from app.ml.bundle import GnnSnapshotScorer, ModelBundle
from app.ml.features import ALL_FEATURES, FEATURE_LABELS, GRAPH_FEATURES, TABULAR_FEATURES, FeatureState, \
    build_feature_state
from app.models import Account, Feedback, ModelVersion, Transaction

log = logging.getLogger(__name__)
PATTERNS = ("ring", "ato", "card_testing")


@dataclass
class TrainResult:
    version: ModelVersion
    bundle: ModelBundle
    accounts: pd.DataFrame
    txns: pd.DataFrame
    X: pd.DataFrame
    state: FeatureState
    gnn_scorer: GnnSnapshotScorer


def load_frames(session: Session) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = session.connection()
    accounts = pd.read_sql(select(Account), conn)
    txns = pd.read_sql(select(Transaction).order_by(Transaction.timestamp, Transaction.id), conn)
    accounts["created_at"] = pd.to_datetime(accounts["created_at"])
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns["is_fraud"] = txns["is_fraud"].astype(bool)
    return accounts, txns.reset_index(drop=True)


def load_feedback(session: Session) -> pd.DataFrame:
    fb = pd.read_sql(select(Feedback).order_by(Feedback.created_at), session.connection())
    # latest verdict wins if an alert was reviewed more than once
    return fb.drop_duplicates("transaction_id", keep="last")


def feedback_known_fraud(fb: pd.DataFrame) -> Dict[str, datetime]:
    confirmed = fb[fb["label"] == "fraud"]
    return {acc: pd.Timestamp(ts).to_pydatetime() for acc, ts in confirmed.groupby("account_id")["created_at"].min().items()}


def _xgb(y: np.ndarray) -> xgb.XGBClassifier:
    pos = max(1, int(y.sum()))
    return xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8,
        min_child_weight=2, scale_pos_weight=(len(y) - pos) / pos, eval_metric="aucpr",
        tree_method="hist", random_state=SEED, n_jobs=4,
    )


def _at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> np.ndarray:
    k = max(1, int(math.ceil(budget * len(s))))
    flagged = np.zeros(len(s), dtype=bool)
    flagged[np.argsort(-s, kind="stable")[:k]] = True
    return flagged


def _metrics(y: np.ndarray, s: np.ndarray, patterns: np.ndarray, budget: float) -> dict:
    flagged = _at_budget(y, s, budget)
    tp = int((flagged & (y == 1)).sum())
    out = {
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "recall_at_budget": round(tp / max(1, int(y.sum())), 4),
        "precision_at_budget": round(tp / max(1, int(flagged.sum())), 4),
        "recall_by_pattern": {},
    }
    for p in PATTERNS:
        m = patterns == p
        if m.any():
            out["recall_by_pattern"][p] = round(float(flagged[m].mean()), 4)
    return out


def train_new_version(session: Session, notes: str = "") -> TrainResult:
    started = time.time()
    accounts, txns = load_frames(session)
    fb = load_feedback(session)
    state, X = build_feature_state(accounts, txns, LABEL_DELAY_DAYS, extra_known=feedback_known_fraud(fb))
    log.info("features: %d rows in %.1fs", len(X), time.time() - started)

    y = txns["is_fraud"].astype(int).to_numpy()
    fb_label = dict(zip(fb["transaction_id"], (fb["label"] == "fraud").astype(int)))
    in_fb = txns["id"].isin(fb_label).to_numpy()
    y[in_fb] = txns.loc[in_fb, "id"].map(fb_label).to_numpy()

    ts = txns["timestamp"]
    train_end, val_end = ts.quantile(0.7).floor("s"), ts.quantile(0.8).floor("s")
    train = ((ts < train_end).to_numpy()) | in_fb
    val = ((ts >= train_end) & (ts < val_end)).to_numpy() & ~in_fb
    test = (ts >= val_end).to_numpy() & ~in_fb
    patterns = txns["pattern"].to_numpy()

    # --- supervised: tabular baseline vs graph-augmented
    tab = _xgb(y[train]).fit(X.loc[train, TABULAR_FEATURES], y[train])
    full = _xgb(y[train]).fit(X.loc[train, ALL_FEATURES], y[train])

    # --- unsupervised anomaly detection
    X_filled = X[ALL_FEATURES].fillna(-1)
    iforest = IsolationForest(n_estimators=250, random_state=SEED, n_jobs=4).fit(X_filled[train].to_numpy())
    anomaly_ref = -iforest.score_samples(X_filled[train].to_numpy())

    # --- GNN on the account graph snapshot at the end of the training window
    snap = build_snapshot(txns, accounts, train_end.to_pydatetime())
    train_labels = pd.Series(y[train], index=txns.loc[train, "account_id"]).groupby(level=0).max().to_dict()
    gnn = GraphRiskModel(seed=SEED).fit(snap, train_labels)
    gnn_scorer = GnnSnapshotScorer(gnn, txns, accounts)
    log.info("models trained (%s backend) in %.1fs", gnn.backend, time.time() - started)

    bundle = ModelBundle(0, full, iforest, anomaly_ref, gnn, meta={})

    def components(mask: np.ndarray) -> Dict[str, np.ndarray]:
        Xm = X.loc[mask]
        return {
            "tabular": tab.predict_proba(Xm[TABULAR_FEATURES])[:, 1],
            "xgb": full.predict_proba(Xm[ALL_FEATURES])[:, 1],
            "anomaly": bundle.anomaly_score(Xm),
            "gnn": gnn_scorer.score(txns.loc[mask]),
        }

    # --- tune ensemble weights on validation (PR-AUC)
    cv, yv = components(val), y[val]
    grid = []
    for wx in np.arange(0.4, 1.0001, 0.05):
        for wg in np.arange(0.0, 1.0001 - wx, 0.05):
            w = {"xgb": round(wx, 2), "gnn": round(wg, 2), "anomaly": round(max(0.0, 1 - wx - wg), 2)}
            s = w["xgb"] * cv["xgb"] + w["gnn"] * cv["gnn"] + w["anomaly"] * cv["anomaly"]
            grid.append((average_precision_score(yv, s), w))
    best_ap = max(ap for ap, _ in grid)
    # among near-equivalent blends, prefer the most diversified one (more robust to drift)
    weights = min((w for ap, w in grid if ap >= best_ap - 0.005), key=lambda w: max(w.values()))
    ens_val = weights["xgb"] * cv["xgb"] + weights["gnn"] * cv["gnn"] + weights["anomaly"] * cv["anomaly"]
    # alert threshold: fill the analyst review budget, and lower it if needed to catch ~90% of validation fraud
    recall_cut = float(np.quantile(ens_val[yv == 1], 0.10)) if yv.sum() else 1.0
    alert_th = min(recall_cut, float(np.quantile(ens_val, 1 - ALERT_BUDGET)))
    alerted = ens_val[ens_val >= alert_th]
    thresholds = {
        "alert": alert_th,
        "high": float(np.quantile(alerted, 0.4)) if len(alerted) else alert_th,
        "critical": float(np.quantile(alerted, 0.75)) if len(alerted) else alert_th,
    }

    # --- evaluate everything on the held-out test window at equal alert volume
    ct, yt, pt = components(test), y[test], patterns[test]
    ct["ensemble"] = weights["xgb"] * ct["xgb"] + weights["gnn"] * ct["gnn"] + weights["anomaly"] * ct["anomaly"]
    models = {name: _metrics(yt, ct[key], pt, ALERT_BUDGET) for name, key in (
        ("tabular_xgboost", "tabular"), ("graph_xgboost", "xgb"), ("isolation_forest", "anomaly"),
        ("graphsage_gnn", "gnn"), ("ensemble", "ensemble"))}
    flag_tab = _at_budget(yt, ct["tabular"], ALERT_BUDGET)
    flag_ens = _at_budget(yt, ct["ensemble"], ALERT_BUDGET)
    hidden = flag_ens & ~flag_tab & (yt == 1)
    amounts = txns.loc[test, "amount"].to_numpy()
    at_threshold = ct["ensemble"] >= thresholds["alert"]

    # --- global explainability: mean |SHAP| on a test sample
    rng = np.random.default_rng(SEED)
    test_idx = np.flatnonzero(test)
    sample = X.iloc[rng.choice(test_idx, size=min(2000, len(test_idx)), replace=False)]
    mean_abs = np.abs(bundle.shap_values(sample)).mean(axis=0)
    importance = sorted(
        ({"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(float(v), 4),
          "group": "graph" if f in GRAPH_FEATURES else "behavioral"} for f, v in zip(ALL_FEATURES, mean_abs)),
        key=lambda d: -d["importance"],
    )[:15]

    version_id = (session.scalar(select(ModelVersion.id).order_by(ModelVersion.id.desc()).limit(1)) or 0) + 1
    metrics = {
        "alert_budget": ALERT_BUDGET,
        "models": models,
        "hidden_fraud": {
            "count": int(hidden.sum()),
            "amount": round(float(amounts[hidden].sum()), 2),
            "by_pattern": {p: int((hidden & (pt == p)).sum()) for p in PATTERNS},
            "test_fraud_total": int(yt.sum()),
        },
        "production_threshold": {
            "alert_rate": round(float(at_threshold.mean()), 4),
            "precision": round(float(yt[at_threshold].mean()) if at_threshold.any() else 0.0, 4),
            "recall": round(float(at_threshold[yt == 1].mean()) if yt.sum() else 0.0, 4),
        },
        "split": {
            "train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum()),
            "train_end": train_end.isoformat(), "test_start": val_end.isoformat(),
            "fraud_rate": round(float(y.mean()), 4),
        },
        "weights": weights,
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "train_seconds": round(time.time() - started, 1),
    }
    bundle.version = version_id
    bundle.meta = {"weights": weights, "thresholds": thresholds, "features": ALL_FEATURES,
                   "gnn_backend": gnn.backend, "trained_at": datetime.utcnow().isoformat()}
    bundle.save()

    session.execute(update(ModelVersion).values(is_active=False))
    mv = ModelVersion(
        id=version_id, created_at=datetime.utcnow(), is_active=True, metrics=metrics,
        feature_importance=importance, n_train=int(train.sum()), n_feedback=int(in_fb.sum()),
        gnn_backend=gnn.backend, notes=notes,
    )
    session.add(mv)
    session.commit()
    log.info("model v%d trained in %.1fs: ensemble PR-AUC %.3f vs tabular %.3f", version_id,
             time.time() - started, models["ensemble"]["pr_auc"], models["tabular_xgboost"]["pr_auc"])
    return TrainResult(mv, bundle, accounts, txns, X, state, gnn_scorer)

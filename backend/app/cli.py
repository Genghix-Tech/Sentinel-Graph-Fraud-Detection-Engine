"""Command line: ``python -m app.cli seed`` builds the demo end to end.

seed  - generate the synthetic world, train model v1, score the recent window, raise
        alerts and backfill analyst reviews for alerts older than a week.
train - train a new model version from current data + analyst feedback.
"""
import argparse
import logging
import shutil
import time
from datetime import timedelta

import pandas as pd
from sqlalchemy import update

from app.config import ALERT_WINDOW_DAYS, ARTIFACT_DIR, SEED
from app.data.generator import generate_world
from app.db import SessionLocal, init_db
from app.ml.pipeline import TrainResult, train_new_version
from app.models import Account, Alert, Feedback, Transaction
from app.services.alerts import build_alerts

log = logging.getLogger("sentinelgraph")


def score_recent_window(session, result: TrainResult) -> int:
    txns, X, bundle = result.txns, result.X, result.bundle
    cutoff = txns["timestamp"].max() - timedelta(days=ALERT_WINDOW_DAYS)
    mask = (txns["timestamp"] >= cutoff).to_numpy()
    sub, Xs = txns[mask].reset_index(drop=True), X[mask].reset_index(drop=True)
    comps = bundle.score(Xs, result.gnn_scorer.score(sub))
    session.execute(update(Transaction), [
        {"id": i, "risk_score": round(float(s), 4), "model_version": bundle.version}
        for i, s in zip(sub["id"], comps["ensemble"])
    ])
    home = dict(zip(result.accounts["id"], result.accounts["country"]))
    alerts = build_alerts(bundle, sub, Xs, comps, home)
    session.add_all(alerts)
    session.commit()
    return len(alerts)


def backfill_reviews(session, older_than_days: int = 7) -> int:
    """Simulate the analyst team having already worked last week's queue."""
    newest = session.query(Transaction.timestamp).order_by(Transaction.timestamp.desc()).first()[0]
    cutoff = newest - timedelta(days=older_than_days)
    rows = (session.query(Alert, Transaction.is_fraud)
            .join(Transaction, Transaction.id == Alert.transaction_id)
            .filter(Alert.created_at < cutoff).all())
    for alert, is_fraud in rows:
        alert.status = "confirmed_fraud" if is_fraud else "dismissed"
        session.add(Feedback(alert_id=alert.id, transaction_id=alert.transaction_id, account_id=alert.account_id,
                             label="fraud" if is_fraud else "legit", analyst="analyst-team",
                             notes="Reviewed in weekly queue", created_at=alert.created_at + timedelta(hours=20)))
    session.commit()
    return len(rows)


def seed(n_accounts: int) -> None:
    started = time.time()
    init_db(drop=True)
    shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)
    frames = generate_world(SEED, n_accounts=n_accounts)
    accounts, txns = frames["accounts"], frames["transactions"]
    log.info("generated %d accounts, %d transactions (%.1f%% fraud)", len(accounts), len(txns),
             100 * txns["is_fraud"].mean())
    with SessionLocal() as s:
        s.bulk_insert_mappings(Account, accounts.to_dict("records"))
        records = txns.to_dict("records")
        for r in records:
            r["timestamp"] = pd.Timestamp(r["timestamp"]).to_pydatetime()
        s.bulk_insert_mappings(Transaction, records)
        s.commit()
        result = train_new_version(s, notes="Initial model trained on chargeback labels")
        n_alerts = score_recent_window(s, result)
        n_reviewed = backfill_reviews(s)
    m = result.version.metrics
    log.info("raised %d alerts (%d already reviewed) in %.0fs", n_alerts, n_reviewed, time.time() - started)
    log.info("test PR-AUC  tabular=%.3f  graph=%.3f  ensemble=%.3f | hidden fraud found: %d",
             m["models"]["tabular_xgboost"]["pr_auc"], m["models"]["graph_xgboost"]["pr_auc"],
             m["models"]["ensemble"]["pr_auc"], m["hidden_fraud"]["count"])


def train(notes: str) -> None:
    with SessionLocal() as s:
        result = train_new_version(s, notes=notes)
    log.info("trained model v%d", result.version.id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="sentinelgraph")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed")
    p_seed.add_argument("--accounts", type=int, default=2500)
    p_train = sub.add_parser("train")
    p_train.add_argument("--notes", default="Manual retrain")
    args = parser.parse_args()
    if args.cmd == "seed":
        seed(args.accounts)
    else:
        train(args.notes)


if __name__ == "__main__":
    main()

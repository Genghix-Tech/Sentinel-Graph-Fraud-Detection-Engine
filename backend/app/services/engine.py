"""Real-time scoring engine.

Holds the incrementally-updated entity graph / feature state, the active model bundle and
the latest GNN account scores. New transactions are featurised against the live graph,
scored, persisted and (if risky) turned into explained alerts in a single request.
"""
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import LABEL_DELAY_DAYS
from app.db import SessionLocal
from app.graph.gnn import build_snapshot
from app.ml.bundle import ModelBundle
from app.ml.features import ALL_FEATURES, FeatureState, build_feature_state
from app.ml.pipeline import feedback_known_fraud, load_feedback, load_frames
from app.models import Account, Alert, ModelVersion, Transaction
from app.services.alerts import build_alerts

log = logging.getLogger(__name__)


class Engine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.ready = False
        self.state: Optional[FeatureState] = None
        self.bundle: Optional[ModelBundle] = None
        self.accounts = pd.DataFrame()
        self.txns = pd.DataFrame()
        self.gnn_scores: Dict[str, float] = {}
        self.last_ts: Optional[datetime] = None
        self._since_gnn_refresh = 0

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        started = time.time()
        with SessionLocal() as s:
            accounts, txns = load_frames(s)
            fb = load_feedback(s)
            active = s.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
        state, _ = build_feature_state(accounts, txns, LABEL_DELAY_DAYS, feedback_known_fraud(fb))
        with self.lock:
            self.accounts, self.txns, self.state = accounts, txns, state
            self.last_ts = txns["timestamp"].max().to_pydatetime() if len(txns) else None
            self.bundle = ModelBundle.load(active.id) if active else None
            self.refresh_gnn()
            self.ready = self.bundle is not None
        log.info("engine loaded %d transactions in %.1fs (model v%s)", len(txns), time.time() - started,
                 self.bundle.version if self.bundle else "-")

    def activate(self, version: int) -> None:
        with self.lock:
            self.bundle = ModelBundle.load(version)
            self.refresh_gnn()
            self.ready = True

    def refresh_gnn(self) -> None:
        """Re-run the GNN over the current graph snapshot (periodic batch job)."""
        if self.bundle is None or self.txns.empty:
            return
        snap = build_snapshot(self.txns, self.accounts, self.txns["timestamp"].max().to_pydatetime())
        self.gnn_scores = self.bundle.gnn.predict(snap)
        self._since_gnn_refresh = 0

    def _gnn_score(self, account_id: str) -> float:
        """GNN risk for an account; accounts newer than the last snapshot borrow their cluster's risk."""
        if account_id in self.gnn_scores:
            return self.gnn_scores[account_id]
        known = [self.gnn_scores[a] for a in self.state.graph.cluster(account_id) if a in self.gnn_scores]
        return float(sum(known) / len(known)) if known else self.bundle.gnn.prior

    def mark_fraud(self, account_id: str, when: datetime) -> None:
        with self.lock:
            if self.state:
                self.state.graph.mark_fraud_known(account_id, when)

    # -------------------------------------------------------------- ingest
    def ingest(self, session: Session, txns: List[dict], new_accounts: List[dict] = ()) -> Tuple[List[Transaction], List[Alert]]:
        """Score and persist transactions (in timestamp order). Commits the session."""
        if not self.ready:
            raise RuntimeError("engine not ready: run `python -m app.cli seed` first")
        with self.lock:
            for acc in new_accounts:
                session.add(Account(**acc))
                self.state.register_account(acc["id"], acc["created_at"], acc["kyc_level"], acc["country"])
            if new_accounts:
                self.accounts = pd.concat([self.accounts, pd.DataFrame(new_accounts)], ignore_index=True)

            rows, feats = [], []
            for t in sorted(txns, key=lambda r: r["timestamp"]):
                if self.last_ts and t["timestamp"] <= self.last_ts:
                    t = {**t, "timestamp": self.last_ts + pd.Timedelta(milliseconds=1)}
                self.last_ts = t["timestamp"]
                feats.append(self.state.process(t))
                rows.append(t)
            frame = pd.DataFrame(rows)
            X = pd.DataFrame(feats, columns=ALL_FEATURES)
            gnn = [self._gnn_score(a) for a in frame["account_id"]]
            comps = self.bundle.score(X, gnn)

            created = []
            for i, t in enumerate(rows):
                fields = {k: t[k] for k in ("id", "account_id", "timestamp", "amount", "merchant_id", "merchant_category",
                                            "channel", "device_id", "ip", "ip_country", "card_id", "address_id")}
                txn = Transaction(**fields, is_fraud=bool(t.get("is_fraud", False)), pattern=t.get("pattern", "unknown"),
                                  risk_score=round(float(comps["ensemble"][i]), 4), model_version=self.bundle.version)
                session.add(txn)
                created.append(txn)
            session.flush()
            home = dict(zip(self.accounts["id"], self.accounts["country"]))
            alerts = build_alerts(self.bundle, frame, X, comps, home)
            session.add_all(alerts)
            session.commit()

            frame["risk_score"] = comps["ensemble"]
            frame["is_fraud"] = frame.get("is_fraud", False)
            self.txns = pd.concat([self.txns, frame[self.txns.columns.intersection(frame.columns)]], ignore_index=True)
            self._since_gnn_refresh += len(rows)
            if self._since_gnn_refresh >= 40:
                self.refresh_gnn()
            return created, alerts


engine = Engine()

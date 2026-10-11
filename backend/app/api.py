"""REST API consumed by the React dashboard (and usable by other services)."""
import logging
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal, get_session
from app.graph.entity_graph import node
from app.ml.pipeline import train_new_version
from app.models import Account, Alert, Feedback, ModelVersion, Transaction
from app.schemas import FeedbackIn, SimulateIn, StreamIn, TransactionIn
from app.services.engine import engine
from app.services.investigation import build_timeline
from app.services.simulator import run_scenario

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

stream_state = {"enabled": False}
retrain_state = {"running": False, "started_at": None, "finished_at": None, "error": None, "version": None}


def _require_engine() -> None:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready. Seed the database with `python -m app.cli seed`.")


def _txn_dict(t: Transaction) -> dict:
    return {
        "id": t.id, "account_id": t.account_id, "timestamp": t.timestamp, "amount": t.amount,
        "merchant_id": t.merchant_id, "merchant_category": t.merchant_category, "channel": t.channel,
        "device_id": t.device_id, "ip": t.ip, "ip_country": t.ip_country, "card_id": t.card_id,
        "address_id": t.address_id, "risk_score": t.risk_score, "model_version": t.model_version,
    }


def _alert_summary(a: Alert, t: Transaction) -> dict:
    top = next((f for f in a.factors if f["contribution"] > 0), None)
    return {
        "id": a.id, "transaction_id": a.transaction_id, "account_id": a.account_id, "created_at": a.created_at,
        "risk_score": a.risk_score, "severity": a.severity, "status": a.status, "scores": a.scores,
        "model_version": a.model_version, "amount": t.amount, "merchant_category": t.merchant_category,
        "channel": t.channel, "top_factor": top["description"] if top else None,
        "graph_driven": any(f["group"] == "graph" for f in a.factors[:3] if f["contribution"] > 0),
    }


# ------------------------------------------------------------------ health
@router.get("/health")
def health():
    return {"status": "ok", "engine_ready": engine.ready,
            "model_version": engine.bundle.version if engine.bundle else None}


# ---------------------------------------------------------------- overview
@router.get("/overview")
def overview(session: Session = Depends(get_session)):
    _require_engine()
    total = session.scalar(select(func.count(Transaction.id)))
    scored = session.scalar(select(func.count(Transaction.id)).where(Transaction.risk_score.is_not(None)))
    rows = session.execute(
        select(Alert.id, Alert.created_at, Alert.severity, Alert.status, Alert.account_id, Transaction.amount)
        .join(Transaction, Transaction.id == Alert.transaction_id)
    ).all()
    alerts = pd.DataFrame(rows, columns=["id", "created_at", "severity", "status", "account_id", "amount"])
    open_ = alerts[alerts["status"] == "open"]
    confirmed = alerts[alerts["status"] == "confirmed_fraud"]

    daily = []
    if len(alerts):
        alerts["day"] = pd.to_datetime(alerts["created_at"]).dt.strftime("%Y-%m-%d")
        pivot = alerts.pivot_table(index="day", columns="severity", values="id", aggfunc="count", fill_value=0)
        for day, r in pivot.iterrows():
            daily.append({"day": day, **{s: int(r.get(s, 0)) for s in ("medium", "high", "critical")}})

    scores = np.array(session.scalars(select(Transaction.risk_score).where(Transaction.risk_score.is_not(None))).all())
    hist, edges = np.histogram(scores, bins=20, range=(0, 1)) if len(scores) else (np.zeros(20), np.linspace(0, 1, 21))

    # rings: clusters of linked accounts with open alerts
    clusters = {}
    with engine.lock:
        g = engine.state.graph
        latest_alert = open_.groupby("account_id")["id"].max().to_dict()
        for acc, n_alerts in Counter(open_["account_id"]).items():
            members = g.cluster(acc)
            if len(members) < 3:
                continue
            key = min(members)
            c = clusters.setdefault(key, {"id": key, "size": len(members), "open_alerts": 0, "accounts": set(),
                                          "known_fraud": g.known_fraud(members, datetime.utcnow())})
            c["open_alerts"] += n_alerts
            c["accounts"].add(acc)
            c["top_alert_id"] = max(c.get("top_alert_id", 0), int(latest_alert[acc]))
    hot = sorted(clusters.values(), key=lambda c: (-c["open_alerts"], -c["size"]))[:6]
    for c in hot:
        c["flagged_accounts"] = len(c.pop("accounts"))

    active = session.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    recent = session.scalars(select(Transaction).order_by(Transaction.timestamp.desc()).limit(12)).all()
    return {
        "kpis": {
            "transactions": total, "scored": scored, "alerts_total": len(alerts), "open_alerts": len(open_),
            "open_by_severity": {s: int((open_["severity"] == s).sum()) for s in ("critical", "high", "medium")},
            "confirmed_fraud": len(confirmed), "dismissed": int((alerts["status"] == "dismissed").sum()),
            "amount_at_risk": round(float(open_["amount"].sum()), 2),
            "amount_prevented": round(float(confirmed["amount"].sum()), 2),
        },
        "daily_alerts": daily,
        "score_histogram": [{"bucket": round(float(edges[i]), 2), "count": int(hist[i])} for i in range(20)],
        "hot_clusters": hot,
        "recent_transactions": [_txn_dict(t) for t in recent],
        "model": None if not active else {
            "version": active.id, "created_at": active.created_at, "gnn_backend": active.gnn_backend,
            "metrics": active.metrics, "n_feedback": active.n_feedback,
        },
        "stream": stream_state,
    }


# ------------------------------------------------------------------ alerts
@router.get("/alerts")
def list_alerts(
    status: Optional[str] = Query(None, pattern="^(open|confirmed_fraud|dismissed)$"),
    severity: Optional[str] = Query(None, pattern="^(critical|high|medium)$"),
    account_id: Optional[str] = None,
    sort: str = Query("recent", pattern="^(recent|risk)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    q = select(Alert, Transaction).join(Transaction, Transaction.id == Alert.transaction_id)
    if status:
        q = q.where(Alert.status == status)
    if severity:
        q = q.where(Alert.severity == severity)
    if account_id:
        q = q.where(Alert.account_id == account_id)
    total = session.scalar(select(func.count()).select_from(q.subquery()))
    order = Alert.risk_score.desc() if sort == "risk" else Alert.created_at.desc()
    rows = session.execute(q.order_by(order, Alert.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "items": [_alert_summary(a, t) for a, t in rows]}


def _get_alert(session: Session, alert_id: int):
    alert = session.get(Alert, alert_id)
    if not alert:
        raise HTTPException(404, "alert not found")
    return alert, session.get(Transaction, alert.transaction_id)


@router.get("/alerts/{alert_id}")
def get_alert(alert_id: int, session: Session = Depends(get_session)):
    _require_engine()
    alert, txn = _get_alert(session, alert_id)
    account = session.get(Account, txn.account_id)
    with engine.lock:
        timeline = build_timeline(session, engine.state.graph, alert, txn)
        cluster = engine.state.graph.cluster(txn.account_id)
    feedback = session.scalars(select(Feedback).where(Feedback.alert_id == alert.id).order_by(Feedback.created_at)).all()
    return {
        **_alert_summary(alert, txn),
        "factors": alert.factors,
        "narrative": alert.narrative,
        "transaction": _txn_dict(txn),
        "account": {"id": account.id, "created_at": account.created_at, "country": account.country,
                    "kyc_level": account.kyc_level, "email_domain": account.email_domain},
        "cluster_size": len(cluster),
        "timeline": timeline,
        "feedback": [{"label": f.label, "notes": f.notes, "analyst": f.analyst, "created_at": f.created_at}
                     for f in feedback],
        # Synthetic data only: lets reviewers of the demo see what the generator injected.
        "ground_truth": {"pattern": txn.pattern, "is_fraud": txn.is_fraud},
    }


@router.get("/alerts/{alert_id}/graph")
def alert_graph(alert_id: int, session: Session = Depends(get_session)):
    _require_engine()
    alert, txn = _get_alert(session, alert_id)
    with engine.lock:
        sub = engine.state.graph.investigation_subgraph(txn.account_id, as_of=datetime.utcnow())
    accounts = [n["label"] for n in sub["nodes"] if n["kind"] == "account"]
    flagged = Counter(session.scalars(select(Alert.account_id).where(Alert.account_id.in_(accounts))).all())
    for n in sub["nodes"]:
        if n["kind"] == "account":
            n["alerts"] = flagged.get(n["label"], 0)
    focus = {node("device", txn.device_id), node("ip", txn.ip), node("card", txn.card_id), node("address", txn.address_id)}
    for n in sub["nodes"]:
        n["in_transaction"] = n["id"] in focus
    return sub


@router.post("/alerts/{alert_id}/feedback")
def submit_feedback(alert_id: int, body: FeedbackIn, session: Session = Depends(get_session)):
    alert, txn = _get_alert(session, alert_id)
    now = datetime.utcnow()
    alert.status = "confirmed_fraud" if body.label == "fraud" else "dismissed"
    session.add(Feedback(alert_id=alert.id, transaction_id=alert.transaction_id, account_id=alert.account_id,
                         label=body.label, analyst=body.analyst, notes=body.notes, created_at=now))
    session.commit()
    if body.label == "fraud":
        # confirmed fraud propagates through the graph immediately for live scoring
        engine.mark_fraud(alert.account_id, now)
    pending = session.scalar(select(func.count(Feedback.id)).where(Feedback.created_at > _active_model_time(session)))
    return {"id": alert.id, "status": alert.status, "feedback_since_last_train": pending}


def _active_model_time(session: Session) -> datetime:
    active = session.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    return active.created_at if active else datetime.min


# ------------------------------------------------------------ transactions
@router.get("/transactions/recent")
def recent_transactions(limit: int = Query(30, ge=1, le=200), session: Session = Depends(get_session)):
    rows = session.scalars(select(Transaction).order_by(Transaction.timestamp.desc()).limit(limit)).all()
    return [_txn_dict(t) for t in rows]


@router.post("/transactions")
def score_transaction(body: TransactionIn, session: Session = Depends(get_session)):
    """Real-time scoring: featurise against the live graph, score, persist, alert if risky."""
    _require_engine()
    new_accounts = []
    if session.get(Account, body.account_id) is None:
        new_accounts.append({"id": body.account_id, "created_at": body.timestamp or datetime.utcnow(),
                             "country": body.account_country, "kyc_level": body.account_kyc_level,
                             "email_domain": "unknown", "pattern": "unknown"})
    txn = body.model_dump(exclude={"account_country", "account_kyc_level"})
    txn["id"] = txn["id"] or f"txn_api_{uuid.uuid4().hex[:10]}"
    txn["timestamp"] = txn["timestamp"] or datetime.utcnow()
    txn["merchant_id"] = txn["merchant_id"] or f"m_{body.merchant_category}_api"
    txn["pattern"] = "unknown"
    created, alerts = engine.ingest(session, [txn], new_accounts)
    t = created[0]
    alert = alerts[0] if alerts else None
    return {
        "transaction_id": t.id, "risk_score": t.risk_score, "model_version": t.model_version,
        "decision": "review" if alert else "approve",
        "severity": alert.severity if alert else None,
        "alert_id": alert.id if alert else None,
        "factors": alert.factors[:5] if alert else [],
        "narrative": alert.narrative if alert else None,
    }


# ------------------------------------------------------------------ models
@router.get("/models")
def list_models(session: Session = Depends(get_session)):
    rows = session.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())).all()
    since = session.scalar(select(func.count(Feedback.id)).where(Feedback.created_at > _active_model_time(session)))
    return {
        "feedback_since_last_train": since,
        "retrain": retrain_state,
        "versions": [{"version": m.id, "created_at": m.created_at, "is_active": m.is_active, "metrics": m.metrics,
                      "feature_importance": m.feature_importance, "n_train": m.n_train, "n_feedback": m.n_feedback,
                      "gnn_backend": m.gnn_backend, "notes": m.notes} for m in rows],
    }


def _retrain_job() -> None:
    try:
        with SessionLocal() as s:
            n_fb = s.scalar(select(func.count(Feedback.id)))
            result = train_new_version(s, notes=f"Retrained with {n_fb} analyst labels")
        engine.activate(result.version.id)
        retrain_state["version"] = result.version.id
    except Exception as exc:  # surfaced to the UI
        log.exception("retrain failed")
        retrain_state["error"] = str(exc)
    finally:
        retrain_state["running"] = False
        retrain_state["finished_at"] = datetime.utcnow()


@router.post("/models/retrain", status_code=202)
def retrain():
    if retrain_state["running"]:
        raise HTTPException(409, "a retrain is already running")
    retrain_state.update(running=True, started_at=datetime.utcnow(), finished_at=None, error=None, version=None)
    threading.Thread(target=_retrain_job, daemon=True).start()
    return retrain_state


# --------------------------------------------------------------- simulate
@router.post("/simulate")
def simulate(body: SimulateIn, session: Session = Depends(get_session)):
    _require_engine()
    return run_scenario(engine, session, body.scenario)


@router.get("/stream")
def get_stream():
    return stream_state


@router.post("/stream")
def set_stream(body: StreamIn):
    stream_state["enabled"] = body.enabled
    return stream_state

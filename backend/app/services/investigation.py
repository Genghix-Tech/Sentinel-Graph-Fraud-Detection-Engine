"""Investigation context for an alert: linked-entity graph and a chronological timeline."""
from datetime import datetime, timedelta
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.graph.entity_graph import EntityGraph, node
from app.models import Account, Alert, Feedback, Transaction


def _event(ts, kind: str, title: str, detail: str = "", severity: str = "info", ref: str = "") -> dict:
    return {"timestamp": ts.isoformat(), "kind": kind, "title": title, "detail": detail,
            "severity": severity, "ref": ref}


def build_timeline(session: Session, graph: EntityGraph, alert: Alert, txn: Transaction) -> List[dict]:
    events: List[dict] = []
    account = session.get(Account, txn.account_id)
    if account:
        events.append(_event(account.created_at, "account", f"Account {account.id} opened",
                             f"Country {account.country}, KYC level {account.kyc_level}"))

    a = node("account", txn.account_id)
    if graph.G.has_node(a):
        for kind, key in (("device", txn.device_id), ("ip", txn.ip), ("card", txn.card_id), ("address", txn.address_id)):
            e = node(kind, key)
            if graph.G.has_edge(a, e):
                first_on_acc = graph.G.edges[a, e]["first_seen"]
                first_anywhere = graph.G.nodes[e]["first_seen"]
                if first_anywhere < first_on_acc:
                    events.append(_event(first_anywhere, "entity", f"{kind.title()} {key} first seen on another account",
                                         f"Now shared by {graph.G.degree(e)} accounts", "warning", e))
                events.append(_event(first_on_acc, "entity", f"{kind.title()} {key} first used by this account",
                                     "", "info", e))

    # linked accounts in the cluster: when they appeared and when fraud was confirmed
    cluster = sorted(graph.cluster(txn.account_id) - {txn.account_id})[:15]
    for other in cluster:
        created = graph.account_created.get(other)
        if created:
            events.append(_event(created, "linked", f"Linked account {other} opened", "", "info", other))
        known = graph.fraud_known_at.get(other)
        if known and known <= datetime.utcnow():
            events.append(_event(known, "fraud", f"Linked account {other} confirmed as fraud",
                                 "Chargeback or analyst confirmation", "critical", other))

    window = session.scalars(
        select(Transaction)
        .where(Transaction.account_id == txn.account_id,
               Transaction.timestamp >= txn.timestamp - timedelta(days=3),
               Transaction.timestamp <= txn.timestamp + timedelta(days=1))
        .order_by(Transaction.timestamp).limit(40)
    ).all()
    for t in window:
        if t.id == txn.id:
            continue
        risky = t.risk_score is not None and t.risk_score >= alert.risk_score * 0.8
        events.append(_event(t.timestamp, "transaction", f"${t.amount:,.2f} at {t.merchant_category.replace('_', ' ')}",
                             f"{t.channel} · device {t.device_id} · IP {t.ip} ({t.ip_country})"
                             + (f" · risk {t.risk_score:.0%}" if t.risk_score is not None else ""),
                             "warning" if risky else "info", t.id))

    events.append(_event(txn.timestamp, "alert", f"Flagged: ${txn.amount:,.2f} at {txn.merchant_category.replace('_', ' ')}",
                         f"Risk {alert.risk_score:.0%} · {alert.severity} severity · model v{alert.model_version}",
                         "critical" if alert.severity == "critical" else "warning", txn.id))
    for fb in session.scalars(select(Feedback).where(Feedback.alert_id == alert.id)).all():
        events.append(_event(fb.created_at, "feedback", f"Analyst marked as {fb.label}",
                             fb.notes or f"by {fb.analyst}", "critical" if fb.label == "fraud" else "info"))
    return sorted(events, key=lambda e: e["timestamp"])

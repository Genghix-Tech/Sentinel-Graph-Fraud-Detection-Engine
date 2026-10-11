"""Live traffic and attack injection for demos.

Each scenario generates events with the same synthetic world used for training, then
compresses them into a short window ending *now* and streams them through the engine.
Attacks deliberately reuse infrastructure from historical fraud (devices, drop
addresses, proxies) so the graph can connect them to known bad actors.
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.data.generator import Profile, SyntheticWorld
from app.services.engine import Engine

SCENARIOS = ("legit", "ring", "ato", "card_testing")


def _compress(events: List[dict], start: datetime, end: datetime) -> List[dict]:
    """Map event times into [start, end] preserving order and relative spacing."""
    if not events:
        return events
    ts = np.array([e["timestamp"].timestamp() for e in events])
    lo, hi = ts.min(), ts.max()
    span = (end - start).total_seconds()
    for e, t in zip(events, ts):
        frac = (t - lo) / (hi - lo) if hi > lo else 1.0
        e["timestamp"] = start + timedelta(seconds=frac * span)
    return events


def _historic(engine: Engine, pattern: str, column: str) -> List[str]:
    t = engine.txns
    return list(t.loc[t["pattern"] == pattern, column].unique())


def run_scenario(engine: Engine, session: Session, scenario: str, n_legit: int = 25) -> Dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = np.random.default_rng()
    now = datetime.utcnow()
    world = SyntheticWorld(int(rng.integers(1 << 30)), end=now + timedelta(days=30), id_tag="L" + uuid.uuid4().hex[:5])
    txns: List[dict]

    if scenario == "legit":
        legit = engine.txns[engine.txns["pattern"] == "legit"]
        sample = legit.sample(n=min(n_legit, len(legit)), random_state=int(rng.integers(1 << 30)))
        txns = []
        for row in sample.to_dict("records"):
            new = {k: row[k] for k in ("account_id", "merchant_id", "merchant_category", "channel", "device_id", "ip",
                                       "ip_country", "card_id", "address_id")}
            new.update(id=world.new_id("txn"), amount=round(float(row["amount"] * rng.lognormal(0, 0.3)), 2),
                       timestamp=now - timedelta(seconds=float(rng.uniform(0, 1200))), is_fraud=False, pattern="legit")
            txns.append(new)
        window = timedelta(minutes=20)
    elif scenario == "ring":
        devices = _historic(engine, "ring", "device_id")
        world.fraud_ring(now, size=int(rng.integers(6, 10)), reuse_devices=[str(rng.choice(devices))] if devices else ())
        txns = world.world.transactions
        window = timedelta(hours=3)
    elif scenario == "ato":
        acc = engine.accounts
        history = engine.txns.groupby("account_id").size()
        established = acc[(acc["pattern"] == "legit") & (acc["created_at"] < now - timedelta(days=90))
                          & acc["id"].map(history).fillna(0).ge(8)]
        victim = established.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        vt = engine.txns[engine.txns["account_id"] == victim["id"]]
        profile = Profile(account_id=victim["id"], country=victim["country"], created_at=victim["created_at"],
                          devices=list(vt["device_id"].unique()), home_ip=vt["ip"].iloc[0],
                          address=vt["address_id"].iloc[0], cards=list(vt["card_id"].unique()),
                          spend_factor=float(vt["amount"].median() / 60), category_pref=np.ones(8) / 8,
                          rate_per_day=0.1)
        world.account_takeover(profile, now, _historic(engine, "ato", "address_id") or [world.new_id("addr")])
        txns = world.world.transactions
        window = timedelta(hours=2)
    else:
        devices = _historic(engine, "card_testing", "device_id") or [world.new_id("dev")]
        proxies = _historic(engine, "card_testing", "ip") or [world.new_ip()]
        world.card_testing(now, devices[:3], proxies[:4])
        txns = world.world.transactions
        window = timedelta(minutes=45)

    start = max(now - window, (engine.last_ts or now - window) + timedelta(seconds=1))
    txns = _compress(sorted(txns, key=lambda e: e["timestamp"]), start, max(now, start + timedelta(seconds=len(txns))))
    new_accounts = []
    for a in world.world.accounts:
        first = min(t["timestamp"] for t in txns if t["account_id"] == a["id"]) if any(
            t["account_id"] == a["id"] for t in txns) else now
        new_accounts.append({**a, "created_at": min(pd.Timestamp(a["created_at"]).to_pydatetime(), first)})

    created, alerts = engine.ingest(session, txns, new_accounts)
    if scenario != "legit":
        engine.refresh_gnn()
    return {
        "scenario": scenario,
        "transactions": len(created),
        "alerts": len(alerts),
        "alert_ids": [a.id for a in alerts],
        "max_score": max((t.risk_score or 0 for t in created), default=0.0),
        "accounts": sorted({t.account_id for t in created}),
    }

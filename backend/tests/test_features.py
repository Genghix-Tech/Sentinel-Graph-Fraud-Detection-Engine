from datetime import datetime, timedelta

import pandas as pd

from app.data.generator import generate_world
from app.ml.features import ALL_FEATURES, FeatureState

T0 = datetime(2026, 1, 1, 12)


def _txn(i, account, minutes, device="dev_shared", card=None, amount=50.0):
    return {"id": f"t{i}", "account_id": account, "timestamp": T0 + timedelta(minutes=minutes), "amount": amount,
            "merchant_id": "m", "merchant_category": "grocery", "channel": "web", "device_id": device,
            "ip": f"ip_{account}", "ip_country": "US", "card_id": card or f"card_{account}",
            "address_id": f"addr_{account}"}


def _state():
    s = FeatureState()
    for acc in ("a", "b", "c"):
        s.register_account(acc, T0 - timedelta(days=100), 2, "US")
    return s


def test_graph_features_only_see_the_past():
    s = _state()
    f1 = s.process(_txn(1, "a", 0))
    assert f1["device_accounts"] == 1 and f1["cluster_size"] == 1
    f2 = s.process(_txn(2, "b", 5))
    assert f2["device_accounts"] == 2 and f2["cluster_size"] == 2 and f2["linked_accounts"] == 1
    assert set(ALL_FEATURES) <= set(f2)


def test_known_fraud_respects_label_delay():
    s = _state()
    s.process(_txn(1, "a", 0))
    s.graph.mark_fraud_known("a", T0 + timedelta(days=10))  # chargeback lands 10 days later
    before = s.process(_txn(2, "b", 60))
    after = s.process(_txn(3, "c", 60 * 24 * 11))
    assert before["known_fraud_linked"] == 0
    assert after["known_fraud_linked"] == 1


def test_velocity_and_novelty():
    s = _state()
    s.process(_txn(1, "a", 0, device="d1"))
    f = s.process(_txn(2, "a", 10, device="d2", card="card_x"))
    assert f["txn_count_1h"] == 2 and f["new_device"] == 1 and f["new_card"] == 1
    assert f["distinct_cards_24h"] == 2


def test_generator_injects_all_patterns():
    frames = generate_world(3, end=datetime(2026, 6, 1), n_accounts=300)
    t = frames["transactions"]
    assert set(t["pattern"]) == {"legit", "ring", "ato", "card_testing"}
    assert 0.01 < t["is_fraud"].mean() < 0.3  # fraud event counts are fixed, so tiny worlds skew high
    assert t["timestamp"].is_monotonic_increasing
    assert isinstance(t["timestamp"].iloc[0], pd.Timestamp)

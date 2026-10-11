"""Point-in-time feature engineering.

``FeatureState`` replays transactions in time order. For each one it computes features
using only what was known at that instant, then folds the transaction into the state.
The same object powers batch training and real-time scoring, so there is no
training/serving skew.
"""
import math
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from app.data.generator import CATEGORY_NAMES
from app.graph.entity_graph import EntityGraph

TABULAR_FEATURES: List[str] = [
    "amount", "log_amount", "hour", "is_night", "channel_app", "account_age_days", "kyc_level",
    "history_len", "txn_count_1h", "txn_count_24h", "amount_sum_24h", "amount_vs_median",
    "log_secs_since_last", "new_device", "new_ip", "new_card", "new_address",
    "ip_country_mismatch", "distinct_cards_24h",
] + [f"cat_{c}" for c in CATEGORY_NAMES]

GRAPH_FEATURES: List[str] = [
    "device_accounts", "card_accounts", "address_accounts", "ip_accounts",
    "acct_devices", "acct_cards", "acct_addresses", "linked_accounts",
    "cluster_size", "cluster_young_ratio", "cluster_creation_span_days", "cluster_txn_24h",
    "known_fraud_linked",
]

ALL_FEATURES: List[str] = TABULAR_FEATURES + GRAPH_FEATURES

FEATURE_LABELS: Dict[str, str] = {
    "amount": "Transaction amount", "log_amount": "Transaction amount (log)", "hour": "Hour of day",
    "is_night": "Night-time transaction", "channel_app": "Mobile app channel",
    "account_age_days": "Account age", "kyc_level": "KYC verification level",
    "history_len": "Prior transactions on account", "txn_count_1h": "Transactions in last hour",
    "txn_count_24h": "Transactions in last 24h", "amount_sum_24h": "Amount spent in last 24h",
    "amount_vs_median": "Amount vs. account's typical spend", "log_secs_since_last": "Time since last transaction",
    "new_device": "First use of this device", "new_ip": "First use of this IP", "new_card": "First use of this card",
    "new_address": "New shipping address", "ip_country_mismatch": "IP country differs from account country",
    "distinct_cards_24h": "Distinct cards used in 24h",
    "device_accounts": "Accounts sharing this device", "card_accounts": "Accounts sharing this card",
    "address_accounts": "Accounts sharing this shipping address", "ip_accounts": "Accounts sharing this IP",
    "acct_devices": "Devices used by account", "acct_cards": "Cards used by account",
    "acct_addresses": "Shipping addresses used by account", "linked_accounts": "Directly linked accounts",
    "cluster_size": "Size of linked-account cluster", "cluster_young_ratio": "Share of young accounts in cluster",
    "cluster_creation_span_days": "Sign-up date spread across cluster", "cluster_txn_24h": "Cluster transactions in 24h",
    "known_fraud_linked": "Linked accounts with confirmed fraud",
}
FEATURE_LABELS.update({f"cat_{c}": f"Merchant category: {c.replace('_', ' ')}" for c in CATEGORY_NAMES})


class FeatureState:
    def __init__(self) -> None:
        self.graph = EntityGraph()
        self.kyc: Dict[str, int] = {}
        self.country: Dict[str, str] = {}
        self.ts: Dict[str, List[float]] = defaultdict(list)
        self.amounts: Dict[str, List[float]] = defaultdict(list)
        self.cards: Dict[str, List[tuple]] = defaultdict(list)  # (epoch, card_id)

    # ---------------------------------------------------------------- setup
    def register_accounts(self, accounts: pd.DataFrame) -> None:
        for row in accounts.itertuples(index=False):
            self.register_account(row.id, row.created_at, row.kyc_level, row.country)

    def register_account(self, account_id: str, created_at: datetime, kyc_level: int, country: str) -> None:
        self.graph.add_account(account_id, pd.Timestamp(created_at).to_pydatetime())
        self.kyc[account_id] = int(kyc_level)
        self.country[account_id] = country

    def register_fraud_labels(self, txns: pd.DataFrame, delay_days: int) -> None:
        """Chargebacks: a fraudulent transaction becomes 'known' ``delay_days`` after it happened."""
        fraud = txns[txns["is_fraud"].astype(bool)]
        for acc, ts in fraud.groupby("account_id")["timestamp"].min().items():
            self.graph.mark_fraud_known(acc, pd.Timestamp(ts).to_pydatetime() + timedelta(days=delay_days))

    # ------------------------------------------------------------- compute
    def process(self, txn: Mapping) -> Dict[str, float]:
        """Compute features for ``txn`` as of its timestamp, then add it to the state."""
        acc = txn["account_id"]
        ts: datetime = pd.Timestamp(txn["timestamp"]).to_pydatetime()
        t = ts.timestamp()
        g = self.graph
        amount = float(txn["amount"])
        created = g.account_created.get(acc, ts)
        hist_ts, hist_amt = self.ts[acc], self.amounts[acc]

        f: Dict[str, float] = {
            "amount": amount,
            "log_amount": math.log1p(amount),
            "hour": ts.hour,
            "is_night": float(ts.hour < 6),
            "channel_app": float(txn["channel"] == "app"),
            "account_age_days": max(0.0, (ts - created).total_seconds() / 86400),
            "kyc_level": self.kyc.get(acc, 1),
            "history_len": len(hist_ts),
            "txn_count_1h": len(hist_ts) - bisect_left(hist_ts, t - 3600) + 1,
            "txn_count_24h": len(hist_ts) - bisect_left(hist_ts, t - 86400) + 1,
            "amount_sum_24h": sum(hist_amt[bisect_left(hist_ts, t - 86400):]) + amount,
            "amount_vs_median": amount / max(1.0, float(np.median(hist_amt))) if len(hist_amt) >= 3 else np.nan,
            "log_secs_since_last": math.log1p(max(0.0, t - hist_ts[-1])) if hist_ts else np.nan,
            "new_device": float(not g.has_link(acc, "device", txn["device_id"])),
            "new_ip": float(not g.has_link(acc, "ip", txn["ip"])),
            "new_card": float(not g.has_link(acc, "card", txn["card_id"])),
            "new_address": float(not g.has_link(acc, "address", txn["address_id"])),
            "ip_country_mismatch": float(txn["ip_country"] != self.country.get(acc, txn["ip_country"])),
        }
        recent_cards = {c for (ct, c) in self.cards[acc] if ct >= t - 86400}
        recent_cards.add(txn["card_id"])
        f["distinct_cards_24h"] = len(recent_cards)
        for c in CATEGORY_NAMES:
            f[f"cat_{c}"] = float(txn["merchant_category"] == c)

        # ---- graph features (after linking this transaction into the graph)
        g.add_transaction({**txn, "timestamp": ts})
        f["device_accounts"] = g.entity_accounts("device", txn["device_id"])
        f["card_accounts"] = g.entity_accounts("card", txn["card_id"])
        f["address_accounts"] = g.entity_accounts("address", txn["address_id"])
        f["ip_accounts"] = g.entity_accounts("ip", txn["ip"])
        f["acct_devices"] = len(g.account_entities(acc, "device"))
        f["acct_cards"] = len(g.account_entities(acc, "card"))
        f["acct_addresses"] = len(g.account_entities(acc, "address"))
        f["linked_accounts"] = len(g.linked_accounts(acc))

        cluster = g.cluster(acc)
        f["cluster_size"] = len(cluster)
        created_times = [g.account_created.get(a, ts) for a in cluster]
        f["cluster_young_ratio"] = float(np.mean([(ts - c).days < 30 for c in created_times]))
        f["cluster_creation_span_days"] = (
            (max(created_times) - min(created_times)).total_seconds() / 86400 if len(cluster) > 1 else np.nan
        )
        f["cluster_txn_24h"] = 1 + sum(
            len(self.ts[a]) - bisect_left(self.ts[a], t - 86400) for a in cluster
        )
        f["known_fraud_linked"] = g.known_fraud(cluster, ts, exclude=acc)

        # ---- fold into history
        hist_ts.append(t)
        hist_amt.append(amount)
        self.cards[acc].append((t, txn["card_id"]))
        return f

    def process_many(self, txns: Iterable[Mapping]) -> pd.DataFrame:
        return pd.DataFrame([self.process(t) for t in txns], columns=ALL_FEATURES)


def build_feature_state(accounts: pd.DataFrame, txns: pd.DataFrame, delay_days: int,
                        extra_known: Optional[Dict[str, datetime]] = None):
    """Replay all transactions; returns (state, feature frame aligned with ``txns``)."""
    state = FeatureState()
    state.register_accounts(accounts)
    state.register_fraud_labels(txns, delay_days)
    for acc, when in (extra_known or {}).items():
        state.graph.mark_fraud_known(acc, when)
    records = txns.to_dict("records")
    feats = state.process_many(records)
    feats.index = txns.index
    return state, feats

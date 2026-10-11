"""Turn SHAP contributions into analyst-readable factors and narratives."""
import math
from typing import Callable, Dict, List, Mapping, Optional

from app.ml.features import FEATURE_LABELS, GRAPH_FEATURES


def _n(v: float) -> int:
    return int(round(v))


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.0f} days"


_TEMPLATES: Dict[str, Callable[[float], str]] = {
    "amount": lambda v: f"Amount of ${v:,.2f}",
    "log_amount": lambda v: f"Amount of ${math.expm1(v):,.2f}",
    "hour": lambda v: f"Placed at {_n(v):02d}:00",
    "is_night": lambda v: "Placed between midnight and 6am" if v else "Placed during daytime hours",
    "channel_app": lambda v: "Made in the mobile app" if v else "Made through web checkout",
    "account_age_days": lambda v: f"Account is {v:.0f} day{'s' if _n(v) != 1 else ''} old",
    "kyc_level": lambda v: f"KYC verification level {_n(v)}",
    "history_len": lambda v: f"{_n(v)} prior transactions on this account",
    "txn_count_1h": lambda v: f"{_n(v)} transactions from this account in the last hour",
    "txn_count_24h": lambda v: f"{_n(v)} transactions from this account in 24h",
    "amount_sum_24h": lambda v: f"${v:,.0f} spent by this account in 24h",
    "amount_vs_median": lambda v: f"{v:.1f}x the account's typical transaction",
    "log_secs_since_last": lambda v: f"{_duration(math.expm1(v))} since the previous transaction",
    "new_device": lambda v: "First time this device is used on the account" if v else "Device previously used by this account",
    "new_ip": lambda v: "First time this IP is used on the account" if v else "Familiar IP address",
    "new_card": lambda v: "First time this card is used on the account" if v else "Card previously used by this account",
    "new_address": lambda v: "Ships to a new address" if v else "Ships to a known address",
    "ip_country_mismatch": lambda v: "IP country differs from the account's country" if v else "IP country matches the account",
    "distinct_cards_24h": lambda v: f"{_n(v)} different cards used in 24h",
    "device_accounts": lambda v: f"Device is shared by {_n(v)} accounts",
    "card_accounts": lambda v: f"Card is used by {_n(v)} accounts",
    "address_accounts": lambda v: f"Shipping address is used by {_n(v)} accounts",
    "ip_accounts": lambda v: f"IP address seen on {_n(v)} accounts",
    "acct_devices": lambda v: f"Account has used {_n(v)} devices",
    "acct_cards": lambda v: f"Account has used {_n(v)} cards",
    "acct_addresses": lambda v: f"Account has shipped to {_n(v)} addresses",
    "linked_accounts": lambda v: f"Directly shares a device, card or address with {_n(v)} accounts",
    "cluster_size": lambda v: f"Member of a {_n(v)}-account linked cluster",
    "cluster_young_ratio": lambda v: f"{v:.0%} of accounts in the cluster are under 30 days old",
    "cluster_creation_span_days": lambda v: f"Linked accounts' sign-up dates span {v:.0f} days",
    "cluster_txn_24h": lambda v: f"{_n(v)} transactions across the cluster in 24h",
    "known_fraud_linked": lambda v: f"{_n(v)} linked account{'s' if _n(v) != 1 else ''} with confirmed fraud",
}


def describe(feature: str, value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{FEATURE_LABELS.get(feature, feature)}: no history"
    if feature.startswith("cat_"):
        cat = feature[4:].replace("_", " ")
        return f"Merchant category is {cat}" if value else f"Merchant category is not {cat}"
    fn = _TEMPLATES.get(feature)
    return fn(value) if fn else f"{FEATURE_LABELS.get(feature, feature)}: {value:.2f}"


def factor(feature: str, value: Optional[float], contribution: float) -> dict:
    clean = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
    return {
        "feature": feature,
        "label": FEATURE_LABELS.get(feature, feature),
        "description": describe(feature, clean),
        "value": clean,
        "contribution": round(float(contribution), 4),
        "group": "graph" if feature in GRAPH_FEATURES else "behavioral",
    }


def _lower_first(text: str) -> str:
    # keep acronyms such as "IP" or "KYC" intact
    return text if len(text) > 1 and text[1].isupper() else text[0].lower() + text[1:]


def build_narrative(score: float, severity: Optional[str], factors: List[dict], feats: Mapping[str, float],
                    components: Mapping[str, float], txn: Mapping, home_country: Optional[str]) -> str:
    """Plain-English explanation an analyst can paste into a case file."""
    sentences = []
    level = (severity or "low").upper()
    sentences.append(f"{level} risk ({score:.0%}) on a ${txn['amount']:,.2f} {txn['merchant_category'].replace('_', ' ')} "
                     f"purchase by {txn['account_id']}.")

    cluster = _n(feats.get("cluster_size", 1) or 1)
    known = _n(feats.get("known_fraud_linked", 0) or 0)
    if cluster >= 3:
        shared = [name for name, key in (("device", "device_accounts"), ("card", "card_accounts"),
                                         ("shipping address", "address_accounts")) if (feats.get(key) or 0) >= 2]
        via = f" through a shared {', '.join(shared)}" if shared else ""
        s = f"The account belongs to a {cluster}-account cluster linked{via}"
        young = feats.get("cluster_young_ratio") or 0
        if young >= 0.5:
            s += f"; {young:.0%} of those accounts are less than 30 days old"
        if known:
            s += f", and {known} of them already have confirmed fraud"
        sentences.append(s + ".")
    elif known:
        sentences.append(f"It is linked to {known} account(s) with confirmed fraud.")

    if feats.get("new_device") and feats.get("ip_country_mismatch") and (feats.get("history_len") or 0) >= 5:
        sentences.append(f"An established account suddenly transacted from a never-seen device with an IP in "
                         f"{txn.get('ip_country')} (account country: {home_country}), a classic account-takeover shape.")
    if (feats.get("txn_count_1h") or 0) >= 6 and (feats.get("distinct_cards_24h") or 0) >= 5:
        sentences.append(f"{_n(feats['txn_count_1h'])} transactions in the past hour across "
                         f"{_n(feats['distinct_cards_24h'])} different cards suggest card testing.")
    ratio = feats.get("amount_vs_median")
    if ratio is not None and not (isinstance(ratio, float) and math.isnan(ratio)) and ratio >= 4:
        sentences.append(f"The amount is {ratio:.1f}x this customer's usual spend.")

    drivers = [f["description"] for f in factors if f["contribution"] > 0][:3]
    if drivers:
        sentences.append("Top drivers: " + "; ".join(_lower_first(d) for d in drivers) + ".")
    sentences.append(
        f"Model signals: supervised {components.get('xgb', 0):.0%}, graph neural network "
        f"{components.get('gnn', 0):.0%}, anomaly detector {components.get('anomaly', 0):.0%}."
    )
    return " ".join(sentences)

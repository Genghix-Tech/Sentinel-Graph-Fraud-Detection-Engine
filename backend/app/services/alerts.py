from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

from app.ml.bundle import ModelBundle
from app.ml.explain import build_narrative
from app.models import Alert


def build_alerts(bundle: ModelBundle, txns: pd.DataFrame, X: pd.DataFrame, comps: Dict[str, np.ndarray],
                 home_country: Mapping[str, str]) -> List[Alert]:
    """Create Alert rows (with SHAP factors + narrative) for transactions above the alert threshold."""
    scores = comps["ensemble"]
    flagged = np.flatnonzero(scores >= bundle.thresholds["alert"])
    if len(flagged) == 0:
        return []
    Xf = X.iloc[flagged]
    explanations = bundle.explain(Xf)
    alerts = []
    for j, pos in enumerate(flagged):
        txn = txns.iloc[pos]
        feats = Xf.iloc[j].to_dict()
        components = {k: round(float(comps[k][pos]), 4) for k in ("xgb", "gnn", "anomaly")}
        score = float(scores[pos])
        severity = bundle.severity(score)
        alerts.append(Alert(
            transaction_id=txn["id"],
            account_id=txn["account_id"],
            created_at=pd.Timestamp(txn["timestamp"]).to_pydatetime(),
            risk_score=round(score, 4),
            severity=severity,
            status="open",
            scores=components,
            factors=explanations[j],
            narrative=build_narrative(score, severity, explanations[j], feats, components, txn,
                                      home_country.get(txn["account_id"])),
            model_version=bundle.version,
        ))
    return alerts

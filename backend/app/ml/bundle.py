"""A versioned set of trained models plus the logic to score and explain transactions."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from app.config import ARTIFACT_DIR
from app.graph.gnn import GraphRiskModel, build_snapshot
from app.ml.explain import factor
from app.ml.features import ALL_FEATURES

log = logging.getLogger(__name__)


def version_dir(version: int) -> Path:
    return ARTIFACT_DIR / f"v{version}"


class ModelBundle:
    def __init__(self, version: int, xgb_graph: xgb.XGBClassifier, iforest, anomaly_ref: np.ndarray,
                 gnn: GraphRiskModel, meta: dict):
        self.version = version
        self.xgb_graph = xgb_graph
        self.iforest = iforest
        self.anomaly_ref = np.sort(anomaly_ref)
        self.gnn = gnn
        self.meta = meta
        self._explainer = None

    # ------------------------------------------------------------- persist
    def save(self) -> Path:
        path = version_dir(self.version)
        path.mkdir(parents=True, exist_ok=True)
        self.xgb_graph.save_model(path / "xgb_graph.json")
        joblib.dump(self.iforest, path / "iforest.joblib")
        np.save(path / "anomaly_ref.npy", self.anomaly_ref)
        self.gnn.save(path)
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return path

    @classmethod
    def load(cls, version: int) -> "ModelBundle":
        path = version_dir(version)
        model = xgb.XGBClassifier()
        model.load_model(path / "xgb_graph.json")
        return cls(version, model, joblib.load(path / "iforest.joblib"), np.load(path / "anomaly_ref.npy"),
                   GraphRiskModel.load(path), json.loads((path / "meta.json").read_text()))

    # --------------------------------------------------------------- score
    @property
    def weights(self) -> Dict[str, float]:
        return self.meta["weights"]

    @property
    def thresholds(self) -> Dict[str, float]:
        return self.meta["thresholds"]

    def anomaly_raw(self, X: pd.DataFrame) -> np.ndarray:
        return -self.iforest.score_samples(X[ALL_FEATURES].fillna(-1).to_numpy())

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Percentile of the isolation-forest anomaly score relative to training traffic."""
        raw = self.anomaly_raw(X)
        return np.searchsorted(self.anomaly_ref, raw) / max(1, len(self.anomaly_ref))

    def score(self, X: pd.DataFrame, gnn_scores: np.ndarray) -> Dict[str, np.ndarray]:
        p_xgb = self.xgb_graph.predict_proba(X[ALL_FEATURES])[:, 1]
        p_anom = self.anomaly_score(X)
        p_gnn = np.asarray(gnn_scores, dtype=float)
        w = self.weights
        ensemble = w["xgb"] * p_xgb + w["gnn"] * p_gnn + w["anomaly"] * p_anom
        return {"xgb": p_xgb, "gnn": p_gnn, "anomaly": p_anom, "ensemble": np.clip(ensemble, 0, 1)}

    def severity(self, score: float) -> Optional[str]:
        th = self.thresholds
        if score >= th["critical"]:
            return "critical"
        if score >= th["high"]:
            return "high"
        if score >= th["alert"]:
            return "medium"
        return None

    # ------------------------------------------------------------- explain
    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature SHAP contributions (log-odds) for the supervised model."""
        data = X[ALL_FEATURES]
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.xgb_graph)
            return np.asarray(self._explainer.shap_values(data))
        except Exception as exc:  # pragma: no cover - version mismatches between shap/xgboost
            log.warning("shap.TreeExplainer failed (%s); using XGBoost native TreeSHAP", exc)
            contribs = self.xgb_graph.get_booster().predict(xgb.DMatrix(data), pred_contribs=True)
            return contribs[:, :-1]

    def explain(self, X: pd.DataFrame, top: int = 8) -> List[List[dict]]:
        values = self.shap_values(X)
        out = []
        for row_vals, (_, row) in zip(values, X[ALL_FEATURES].iterrows()):
            order = np.argsort(-np.abs(row_vals))[:top]
            out.append([factor(ALL_FEATURES[j], row.iloc[j], row_vals[j]) for j in order])
        return out


class GnnSnapshotScorer:
    """Scores transactions with the GNN using the graph snapshot at the start of their time bucket."""

    def __init__(self, model: GraphRiskModel, txns: pd.DataFrame, accounts: pd.DataFrame, bucket_hours: int = 6):
        self.model, self.txns, self.accounts = model, txns, accounts
        self.freq = f"{bucket_hours}h"
        self._cache: Dict[pd.Timestamp, Dict[str, float]] = {}

    def predictions_at(self, as_of: datetime) -> Dict[str, float]:
        key = pd.Timestamp(as_of)
        if key not in self._cache:
            self._cache[key] = self.model.predict(build_snapshot(self.txns, self.accounts, key.to_pydatetime()))
        return self._cache[key]

    def score(self, subset: pd.DataFrame) -> np.ndarray:
        out = np.full(len(subset), self.model.prior, dtype=float)
        buckets = subset["timestamp"].dt.floor(self.freq).to_numpy()
        accounts = subset["account_id"].to_numpy()
        for bucket in np.unique(buckets):
            idx = np.flatnonzero(buckets == bucket)
            preds = self.predictions_at(bucket)
            out[idx] = [preds.get(a, self.model.prior) for a in accounts[idx]]
        return out

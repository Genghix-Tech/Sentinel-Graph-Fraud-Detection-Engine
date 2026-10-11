"""Account-level graph neural network.

Accounts become nodes, linked when they share a device, card, address or a small IP.
A GraphSAGE model (PyTorch Geometric) learns account risk from each node's behaviour
*and* its neighbourhood. It runs as a periodic batch job over graph snapshots; fresh
links between snapshots are covered by the real-time graph features.

If PyTorch Geometric isn't installed, an SGC-style fallback (logistic regression on
features propagated over the normalised adjacency, i.e. [X, AX, A^2X]) is used so the
system still runs everywhere.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

HIGH_RISK_CATEGORIES = {"electronics", "gift_cards", "crypto"}
NODE_FEATURES = [
    "log_n_txn", "log_mean_amount", "log_max_amount", "n_devices", "n_cards", "n_addresses", "n_ips",
    "night_ratio", "foreign_ratio", "high_risk_ratio", "log_age_days", "kyc_level", "active_span_days",
]
MAX_IP_SHARE = 10  # IPs shared by more accounts (offices, NAT) are not used as links
MAX_ENTITY_FANOUT = 60


@dataclass
class Snapshot:
    accounts: List[str]
    x: np.ndarray  # (n_nodes, n_features), raw
    edge_index: np.ndarray  # (2, n_edges), undirected (both directions)

    @property
    def index(self) -> Dict[str, int]:
        return {a: i for i, a in enumerate(self.accounts)}


def build_snapshot(txns: pd.DataFrame, accounts: pd.DataFrame, as_of: datetime) -> Snapshot:
    t = txns[txns["timestamp"] <= as_of]
    if t.empty:
        return Snapshot([], np.zeros((0, len(NODE_FEATURES))), np.zeros((2, 0), dtype=np.int64))
    acc = accounts.set_index("id")
    t = t.assign(
        is_night=t["timestamp"].dt.hour < 6,
        high_risk=t["merchant_category"].isin(HIGH_RISK_CATEGORIES),
        foreign=t["ip_country"].to_numpy() != acc["country"].reindex(t["account_id"]).to_numpy(),
    )
    g = t.groupby("account_id")
    df = pd.DataFrame({
        "log_n_txn": np.log1p(g.size()),
        "log_mean_amount": np.log1p(g["amount"].mean()),
        "log_max_amount": np.log1p(g["amount"].max()),
        "n_devices": g["device_id"].nunique(),
        "n_cards": g["card_id"].nunique(),
        "n_addresses": g["address_id"].nunique(),
        "n_ips": g["ip"].nunique(),
        "night_ratio": g["is_night"].mean(),
        "high_risk_ratio": g["high_risk"].mean(),
        "foreign_ratio": g["foreign"].mean(),
        "active_span_days": (g["timestamp"].max() - g["timestamp"].min()).dt.total_seconds() / 86400,
    })
    created = pd.to_datetime(acc["created_at"].reindex(df.index))
    df["log_age_days"] = np.log1p(((pd.Timestamp(as_of) - created).dt.total_seconds() / 86400).clip(lower=0))
    df["kyc_level"] = acc["kyc_level"].reindex(df.index).fillna(1)
    df = df[NODE_FEATURES].fillna(0.0)

    accounts_list = list(df.index)
    node_idx = pd.Series(np.arange(len(accounts_list)), index=accounts_list)
    links = []
    for col, cap in (("device_id", MAX_ENTITY_FANOUT), ("card_id", MAX_ENTITY_FANOUT),
                     ("address_id", MAX_ENTITY_FANOUT), ("ip", MAX_IP_SHARE)):
        members = t[[col, "account_id"]].drop_duplicates()
        sizes = members[col].map(members[col].value_counts())
        members = members[(sizes >= 2) & (sizes <= cap)]
        members = pd.DataFrame({"e": members[col].to_numpy(), "n": node_idx[members["account_id"]].to_numpy()})
        pairs = members.merge(members, on="e")
        links.append(pairs.loc[pairs["n_x"] < pairs["n_y"], ["n_x", "n_y"]].to_numpy())
    e = np.unique(np.concatenate(links), axis=0).T if links and sum(map(len, links)) else np.zeros((2, 0), np.int64)
    edge_index = np.concatenate([e, e[::-1]], axis=1).astype(np.int64)
    return Snapshot(accounts_list, df.to_numpy(dtype=np.float32), edge_index)


def _normalized_adjacency(n: int, edge_index: np.ndarray) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n)).tocsr()
    a = a + sp.eye(n)
    d = np.asarray(a.sum(axis=1)).ravel()
    d_inv = sp.diags(1.0 / np.sqrt(d))
    return d_inv @ a @ d_inv


def torch_geometric_available() -> bool:
    if os.getenv("GNN_BACKEND", "").lower() == "sgc":
        return False
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


class GraphRiskModel:
    """Node classifier producing a fraud probability per account."""

    def __init__(self, backend: Optional[str] = None, seed: int = 7):
        self.backend = backend or ("graphsage" if torch_geometric_available() else "sgc")
        self.seed = seed
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.model = None
        self.prior = 0.0

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit(self, snap: Snapshot, labels: Dict[str, int]) -> "GraphRiskModel":
        y = np.array([labels.get(a, 0) for a in snap.accounts], dtype=np.float32)
        self.mean = snap.x.mean(axis=0)
        self.std = snap.x.std(axis=0) + 1e-6
        self.prior = float(y.mean())
        x = self._scale(snap.x)
        if self.backend == "graphsage":
            self._fit_sage(x, snap.edge_index, y)
        else:
            feats = self._sgc_features(x, snap.edge_index)
            self.model = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(feats, y)
        return self

    def predict(self, snap: Snapshot) -> Dict[str, float]:
        if not snap.accounts:
            return {}
        x = self._scale(snap.x)
        if self.backend == "graphsage":
            p = self._predict_sage(x, snap.edge_index)
        else:
            p = self.model.predict_proba(self._sgc_features(x, snap.edge_index))[:, 1]
        return dict(zip(snap.accounts, p.astype(float)))

    # ---------------------------------------------------------- SGC fallback
    @staticmethod
    def _sgc_features(x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        a = _normalized_adjacency(x.shape[0], edge_index)
        ax = a @ x
        return np.hstack([x, ax, a @ ax])

    # ----------------------------------------------------- PyG GraphSAGE
    def _build_sage(self, in_dim: int):
        import torch
        from torch_geometric.nn import SAGEConv

        class SAGE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, 32)
                self.conv2 = SAGEConv(32, 32)
                self.head = torch.nn.Linear(32 + in_dim, 1)  # skip connection keeps node's own signal

            def forward(self, x, edge_index):
                h = torch.relu(self.conv1(x, edge_index))
                h = torch.nn.functional.dropout(h, p=0.2, training=self.training)
                h = torch.relu(self.conv2(h, edge_index))
                return self.head(torch.cat([h, x], dim=1)).squeeze(-1)

        return SAGE()

    def _fit_sage(self, x: np.ndarray, edge_index: np.ndarray, y: np.ndarray) -> None:
        import torch

        torch.manual_seed(self.seed)
        model = self._build_sage(x.shape[1])
        xt, et, yt = torch.tensor(x), torch.tensor(edge_index), torch.tensor(y)
        pos = max(1.0, float(y.sum()))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for _ in range(250):
            opt.zero_grad()
            loss = loss_fn(model(xt, et), yt)
            loss.backward()
            opt.step()
        model.eval()
        self.model = model

    def _predict_sage(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(x), torch.tensor(edge_index))).numpy()

    # ------------------------------------------------------------ persist
    def save(self, path: Path) -> None:
        state = {"backend": self.backend, "mean": self.mean, "std": self.std, "prior": self.prior,
                 "in_dim": len(NODE_FEATURES)}
        if self.backend == "graphsage":
            import torch
            torch.save(self.model.state_dict(), path / "gnn_sage.pt")
        else:
            state["model"] = self.model
        joblib.dump(state, path / "gnn_meta.joblib")

    @classmethod
    def load(cls, path: Path) -> "GraphRiskModel":
        state = joblib.load(path / "gnn_meta.joblib")
        m = cls(backend=state["backend"])
        m.mean, m.std, m.prior = state["mean"], state["std"], state["prior"]
        if m.backend == "graphsage":
            import torch
            m.model = m._build_sage(state["in_dim"])
            m.model.load_state_dict(torch.load(path / "gnn_sage.pt", weights_only=True))
            m.model.eval()
        else:
            m.model = state["model"]
        return m

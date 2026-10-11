"""Heterogeneous account-entity graph (NetworkX).

Nodes are ``account:<id>``, ``device:<id>``, ``ip:<ip>``, ``card:<id>`` and ``address:<id>``.
Accounts connect only to entities, so two accounts are "linked" when they share an entity.
The graph is built incrementally in timestamp order, which lets every feature be computed
*as of* the moment a transaction happened (no peeking at future links).
"""
from collections import deque
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Set

import networkx as nx

from app.config import STRONG_ENTITY_TYPES

ENTITY_FIELDS = (("device", "device_id"), ("ip", "ip"), ("card", "card_id"), ("address", "address_id"))


def node(kind: str, key: str) -> str:
    return f"{kind}:{key}"


def split_node(n: str):
    kind, _, key = n.partition(":")
    return kind, key


class EntityGraph:
    def __init__(self) -> None:
        self.G = nx.Graph()
        self.account_created: Dict[str, datetime] = {}
        # account -> moment its fraud label became known (chargeback + delay, or analyst feedback)
        self.fraud_known_at: Dict[str, datetime] = {}

    # ----------------------------------------------------------------- build
    def add_account(self, account_id: str, created_at: datetime) -> None:
        self.account_created[account_id] = created_at
        n = node("account", account_id)
        if not self.G.has_node(n):
            self.G.add_node(n, kind="account", first_seen=created_at)

    def add_transaction(self, txn: Mapping) -> None:
        a = node("account", txn["account_id"])
        ts = txn["timestamp"]
        if not self.G.has_node(a):
            self.G.add_node(a, kind="account", first_seen=ts)
        for kind, field in ENTITY_FIELDS:
            e = node(kind, txn[field])
            if not self.G.has_node(e):
                self.G.add_node(e, kind=kind, first_seen=ts)
            if self.G.has_edge(a, e):
                d = self.G.edges[a, e]
                d["count"] += 1
                d["last_seen"] = ts
            else:
                self.G.add_edge(a, e, kind=kind, count=1, first_seen=ts, last_seen=ts)

    def mark_fraud_known(self, account_id: str, when: datetime) -> None:
        prev = self.fraud_known_at.get(account_id)
        if prev is None or when < prev:
            self.fraud_known_at[account_id] = when

    # --------------------------------------------------------------- queries
    def has_link(self, account_id: str, kind: str, key: str) -> bool:
        return self.G.has_edge(node("account", account_id), node(kind, key))

    def entity_accounts(self, kind: str, key: str) -> int:
        n = node(kind, key)
        return self.G.degree(n) if self.G.has_node(n) else 0

    def account_entities(self, account_id: str, kind: str) -> List[str]:
        a = node("account", account_id)
        if not self.G.has_node(a):
            return []
        return [split_node(e)[1] for e in self.G.neighbors(a) if e.startswith(kind + ":")]

    def linked_accounts(self, account_id: str, kinds: Iterable[str] = STRONG_ENTITY_TYPES) -> Set[str]:
        """Accounts sharing at least one entity of ``kinds`` with this account."""
        a = node("account", account_id)
        out: Set[str] = set()
        if not self.G.has_node(a):
            return out
        kinds = tuple(kinds)
        for e in self.G.neighbors(a):
            if split_node(e)[0] in kinds:
                out.update(split_node(x)[1] for x in self.G.neighbors(e) if x != a)
        return out

    def cluster(self, account_id: str, kinds: Iterable[str] = STRONG_ENTITY_TYPES, cap: int = 300) -> Set[str]:
        """Connected component of accounts reachable through shared strong entities."""
        kinds = tuple(kinds)
        seen = {account_id}
        queue = deque([account_id])
        while queue and len(seen) < cap:
            current = queue.popleft()
            for other in self.linked_accounts(current, kinds):
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        return seen

    def known_fraud(self, accounts: Iterable[str], as_of: datetime, exclude: Optional[str] = None) -> int:
        return sum(
            1 for acc in accounts
            if acc != exclude and acc in self.fraud_known_at and self.fraud_known_at[acc] <= as_of
        )

    # ------------------------------------------------------------ for the UI
    def investigation_subgraph(self, account_id: str, as_of: Optional[datetime] = None,
                               max_accounts: int = 40) -> Dict[str, list]:
        """Accounts in this account's cluster, plus every entity they use."""
        accounts = sorted(self.cluster(account_id, cap=max_accounts))
        nodes: Dict[str, dict] = {}
        edges = []
        for acc in accounts:
            a = node("account", acc)
            known = acc in self.fraud_known_at and (as_of is None or self.fraud_known_at[acc] <= as_of)
            nodes[a] = {"id": a, "kind": "account", "label": acc, "focus": acc == account_id,
                        "known_fraud": known}
            for e in self.G.neighbors(a):
                kind, key = split_node(e)
                shared = self.G.degree(e)
                # keep shared entities and the focus account's own entities; drop noise
                if shared < 2 and acc != account_id:
                    continue
                nodes.setdefault(e, {"id": e, "kind": kind, "label": key, "shared_by": shared,
                                     "focus": False, "known_fraud": False})
                d = self.G.edges[a, e]
                edges.append({"source": a, "target": e, "kind": kind, "count": d["count"],
                              "first_seen": d["first_seen"].isoformat()})
        return {"nodes": list(nodes.values()), "edges": edges}

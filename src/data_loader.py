"""
src/data_loader.py
==================
Phase 1 — Single Source of Truth

ONE module that ALL other files import from.
Loads from /data/cleaned/ exclusively.
Caches graphs so they are built only once per process.

Public API
----------
get_email_graph()       -> nx.Graph           # undirected, weighted
get_proximity_graph()   -> nx.Graph           # undirected, weighted
get_monthly_snapshots() -> dict[str, nx.Graph] # keyed by 'YYYY-MM'
"""
from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path



import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
CLEANED = _ROOT / "data" / "cleaned"

EMAIL_AGG     = CLEANED / "email_edges_aggregated_cleaned.csv"
EMAIL_EDGES   = CLEANED / "email_edges_cleaned.csv"
PROXIMITY     = CLEANED / "proximity_edges_cleaned.csv"
NODE_DEPTS    = CLEANED / "node_departments_cleaned.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_files() -> None:
    """Warn about any missing cleaned files."""
    for path in [EMAIL_AGG, EMAIL_EDGES, PROXIMITY, NODE_DEPTS]:
        if not path.exists():
            print(f"WARNING: {path.name} not found in /data/cleaned/. "
                  f"Run scripts/clean_data.py first.")


def _build_email_graph() -> nx.Graph:
    """Build undirected weighted email graph from aggregated cleaned data."""
    if not EMAIL_AGG.exists():
        print(f"WARNING: {EMAIL_AGG.name} missing — returning empty graph.")
        return nx.Graph(name="Email_Layer")

    df = pd.read_csv(EMAIL_AGG)
    G = nx.Graph(name="Email_Layer")

    for row in df.itertuples(index=False):
        u, v, w = row.sender, row.recipient, row.weight
        if G.has_edge(u, v):
            G[u][v]["weight"] += float(w)
        else:
            G.add_edge(u, v, weight=float(w))

    return G


def _infer_email_dept_map() -> dict[str, str]:
    """Build sender → department map (mode per sender) from cleaned email edges."""
    if not EMAIL_EDGES.exists():
        return {}

    # Only load sender + department columns — saves memory on the 88MB file
    df = pd.read_csv(EMAIL_EDGES, usecols=["sender", "department"])

    dept_map: dict[str, str] = (
        df.groupby("sender")["department"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .to_dict()
    )
    return dept_map


def _build_proximity_graph() -> nx.Graph:
    """Build undirected weighted proximity graph."""
    if not PROXIMITY.exists():
        print(f"WARNING: {PROXIMITY.name} missing — returning empty graph.")
        return nx.Graph(name="Proximity_Layer")

    df = pd.read_csv(PROXIMITY)
    G = nx.Graph(name="Proximity_Layer")

    for row in df.itertuples(index=False):
        G.add_edge(row.i, row.j, weight=float(row.weight))

    return G


def _load_node_depts() -> dict:
    """Load node_id → department from node_departments_cleaned.csv."""
    if not NODE_DEPTS.exists():
        return {}
    df = pd.read_csv(NODE_DEPTS)
    return dict(zip(df["node_id"], df["department"]))


def _build_monthly_snapshots() -> dict[str, nx.Graph]:
    """Slice cleaned email edges into per-month undirected graphs."""
    if not EMAIL_EDGES.exists() or not EMAIL_AGG.exists():
        print("WARNING: edges or aggregated edges missing — no snapshots available.")
        return {}

    # Get valid nodes from the top-200 aggregated set
    valid_nodes = set(pd.read_csv(EMAIL_AGG)["sender"]).union(
                  set(pd.read_csv(EMAIL_AGG)["recipient"]))

    df = pd.read_csv(EMAIL_EDGES, usecols=["sender", "recipient", "timestamp"])
    df = df[df["sender"].isin(valid_nodes) & df["recipient"].isin(valid_nodes)]
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["period"] = df["timestamp"].dt.to_period("M")

    snapshots: dict[str, nx.Graph] = {}
    for period, group in df.groupby("period"):
        key = str(period)
        G = nx.Graph(name=f"Email_{key}")
        for row in group.itertuples(index=False):
            u, v = row.sender, row.recipient
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)
        snapshots[key] = G

    return snapshots


# ---------------------------------------------------------------------------
# Cached public API
# ---------------------------------------------------------------------------

_email_graph: nx.Graph | None = None
_proximity_graph: nx.Graph | None = None
_monthly_snapshots: dict[str, nx.Graph] | None = None


def get_email_graph() -> nx.Graph:
    """
    Return cached undirected weighted email graph.
    Nodes carry 'dept' attribute (inferred from most-frequent sender dept).
    Edges carry 'weight' = number of messages between the pair.
    """
    global _email_graph
    if _email_graph is None:
        _check_files()
        _email_graph = _build_email_graph()

        # Attach dept attributes
        dept_map = _infer_email_dept_map()
        attrs = {node: dept_map.get(node, "Unknown") for node in _email_graph.nodes()}
        nx.set_node_attributes(_email_graph, attrs, name="dept")

        # Summary
        depts = set(nx.get_node_attributes(_email_graph, "dept").values())
        ts_min, ts_max = _get_email_date_range()
        print(f"G_email: {_email_graph.number_of_nodes()} nodes, "
              f"{_email_graph.number_of_edges()} edges, "
              f"{len(depts)} depts, "
              f"date range {ts_min} to {ts_max}")

    return _email_graph


def get_proximity_graph() -> nx.Graph:
    """
    Return cached undirected weighted proximity graph.
    Nodes carry 'dept' attribute from node_departments_cleaned.csv.
    Edges carry 'weight' = total contact seconds.
    """
    global _proximity_graph
    if _proximity_graph is None:
        _check_files()
        _proximity_graph = _build_proximity_graph()

        # Attach dept attributes
        node_dept_map = _load_node_depts()
        attrs = {node: node_dept_map.get(node, "Unknown")
                 for node in _proximity_graph.nodes()}
        nx.set_node_attributes(_proximity_graph, attrs, name="dept")

        depts = set(nx.get_node_attributes(_proximity_graph, "dept").values())
        print(f"G_proximity: {_proximity_graph.number_of_nodes()} nodes, "
              f"{_proximity_graph.number_of_edges()} edges, "
              f"{len(depts)} depts")

    return _proximity_graph


def get_monthly_snapshots() -> dict[str, nx.Graph]:
    """
    Return cached dict of per-month undirected graphs (email layer).
    Keys are 'YYYY-MM' strings.
    """
    global _monthly_snapshots
    if _monthly_snapshots is None:
        _monthly_snapshots = _build_monthly_snapshots()
        print(f"Monthly snapshots: {len(_monthly_snapshots)} periods "
              f"({min(_monthly_snapshots) if _monthly_snapshots else 'N/A'} – "
              f"{max(_monthly_snapshots) if _monthly_snapshots else 'N/A'})")
    return _monthly_snapshots


def _get_email_date_range() -> tuple[str, str]:
    """Return (min_date, max_date) from cleaned email edges."""
    if not EMAIL_EDGES.exists():
        return "N/A", "N/A"
    df = pd.read_csv(EMAIL_EDGES, usecols=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna()
    return str(df["timestamp"].min().date()), str(df["timestamp"].max().date())


# ---------------------------------------------------------------------------
# Direct run — smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  data_loader.py — smoke test")
    print("=" * 55)
    G_e = get_email_graph()
    G_p = get_proximity_graph()
    snaps = get_monthly_snapshots()
    print(f"\nAll OK. Snapshots: {list(snaps.keys())[:5]} …")

"""
src/forecasting/feature_engineering.py
========================================
Phase 3 — Feature Engineering for ML

Builds a node-level feature matrix from pre-built graphs and analysis outputs.
No data loading here — all inputs are passed as arguments from data_loader.py.

Run directly:
    python src/forecasting/feature_engineering.py

Saves to: data/features.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import linregress

from src.analysis.homophily import coleman_homophily
from src.analysis.structural_holes import burt_constraint


# ---------------------------------------------------------------------------
# Static feature helpers
# ---------------------------------------------------------------------------

def _weighted_degree(G: nx.Graph, node) -> float:
    """Sum of edge weights for node (weighted degree)."""
    if not G.has_node(node):
        return 0.0
    return float(sum(d.get("weight", 1.0) for _, _, d in G.edges(node, data=True)))


def _clustering(G: nx.Graph, node) -> float:
    if not G.has_node(node):
        return 0.0
    try:
        return float(nx.clustering(G, node, weight="weight"))
    except Exception:
        return 0.0


def _cross_layer_closure(G_email: nx.Graph, G_proximity: nx.Graph, node) -> float:
    """
    Fraction of open triads centred on *node* in G_email that are closed in
    G_proximity.  Returns 0 if node not in email graph or < 2 neighbours.
    NOTE: cross-layer requires nodes to exist in both graphs.  Since email
    nodes are addresses and proximity nodes are integers, this will be 0 for
    most nodes unless an explicit node-mapping is provided.
    """
    if not G_email.has_node(node):
        return 0.0
    nbrs = [n for n in G_email.neighbors(node) if G_proximity.has_node(n)]
    if len(nbrs) < 2:
        return 0.0

    open_triads = 0
    closed_in_prox = 0
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            u, v = nbrs[i], nbrs[j]
            if not G_email.has_edge(u, v):
                open_triads += 1
                if G_proximity.has_edge(u, v):
                    closed_in_prox += 1

    return closed_in_prox / open_triads if open_triads > 0 else 0.0


# ---------------------------------------------------------------------------
# Temporal betweenness feature helpers
# ---------------------------------------------------------------------------

def _tb_features_for_node(node, tb_matrix: pd.DataFrame) -> dict:
    """
    Derive tb_mean, tb_trend, tb_final, tb_drop from the TB matrix row for node.

    tb_matrix: nodes × months DataFrame (output of build_tb_matrix).
    """
    if tb_matrix.empty or node not in tb_matrix.index:
        return {"tb_mean": 0.0, "tb_trend": 0.0, "tb_final": 0.0, "tb_drop": 0.0}

    vals = tb_matrix.loc[node].values.astype(float)
    if len(vals) == 0:
        return {"tb_mean": 0.0, "tb_trend": 0.0, "tb_final": 0.0, "tb_drop": 0.0}

    tb_mean  = float(np.nanmean(vals))
    tb_final = float(vals[-1])

    # Linear trend (slope)
    x = np.arange(len(vals), dtype=float)
    valid = ~np.isnan(vals)
    if valid.sum() >= 2:
        slope, *_ = linregress(x[valid], vals[valid])
        tb_trend = float(slope)
    else:
        tb_trend = 0.0

    # Max single-month drop
    diffs = np.diff(vals)
    drops = diffs[diffs < 0]
    tb_drop = float(-drops.min()) if len(drops) > 0 else 0.0

    return {"tb_mean": tb_mean, "tb_trend": tb_trend,
            "tb_final": tb_final, "tb_drop": tb_drop}


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _build_labels(
    email_df: pd.DataFrame,
    drop_threshold: float = 0.70,
    streak_months: int = 2,
    rolling_window: int = 6,
) -> pd.Series:
    """
    Label nodes as at-risk (1) if their monthly email volume drops >70%
    for 2+ consecutive months vs their own 6-month rolling average.
    Otherwise label 0 (stable).

    Returns
    -------
    pd.Series indexed by node, values in {0, 1}.
    """
    df = email_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")

    vol = (
        df.groupby(["sender", "month"])
        .size()
        .reset_index(name="volume")
        .rename(columns={"sender": "node"})
        .sort_values(["node", "month"])
    )

    at_risk: set[str] = set()

    for node, grp in vol.groupby("node"):
        grp = grp.sort_values("month").reset_index(drop=True)
        vols  = grp["volume"].values.astype(float)

        if len(vols) < rolling_window + streak_months:
            continue

        consecutive_drops = 0
        for i in range(rolling_window, len(vols)):
            roll_avg = vols[max(0, i - rolling_window):i].mean()
            if roll_avg > 0 and (1.0 - vols[i] / roll_avg) > drop_threshold:
                consecutive_drops += 1
                if consecutive_drops >= streak_months:
                    at_risk.add(node)
                    break
            else:
                consecutive_drops = 0

    # All senders
    all_nodes = vol["node"].unique()
    labels = pd.Series(
        {node: (1 if node in at_risk else 0) for node in all_nodes},
        name="label",
    )
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    G_email: nx.Graph,
    G_proximity: nx.Graph,
    tb_matrix: pd.DataFrame,
    email_df: pd.DataFrame,
    drop_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Build a node-level feature matrix.

    Parameters
    ----------
    G_email     : undirected weighted email graph (from data_loader)
    G_proximity : undirected weighted proximity graph (from data_loader)
    tb_matrix   : nodes × periods betweenness matrix (from build_tb_matrix)
    email_df    : cleaned email edges DataFrame (for label construction)
    drop_threshold : volume-drop ratio that triggers at-risk label (default 0.70)

    Returns
    -------
    pd.DataFrame   columns: node | all features | label
    """
    print("  Computing Coleman homophily …")
    h_email = coleman_homophily(G_email, dept_attr="dept", weight_attr="weight")
    h_prox  = coleman_homophily(G_proximity, dept_attr="dept", weight_attr="weight")

    print("  Computing Burt constraint …")
    c_email = burt_constraint(G_email, weight_attr="weight")

    print("  Building labels …")
    labels = _build_labels(email_df, drop_threshold=drop_threshold)
    n_risk   = (labels == 1).sum()
    n_stable = (labels == 0).sum()
    print(f"  Labels: {n_risk} at-risk, {n_stable} stable")

    ratio = max(n_stable, n_risk) / max(min(n_stable, n_risk), 1)
    if ratio > 4.0:
        print(f"  WARNING: class imbalance {ratio:.1f}:1 — will apply SMOTE during model training")

    print("  Assembling feature rows …")
    records: list[dict] = []

    all_nodes = list(G_email.nodes())

    for node in all_nodes:
        feat: dict = {"node": node}

        # Email graph features
        feat["degree"]           = _weighted_degree(G_email, node)
        feat["clustering"]       = _clustering(G_email, node)
        feat["burt_constraint"]  = c_email.get(node, np.nan)
        feat["coleman_homophily"] = h_email.get(node, np.nan)
        feat["cross_closure"]    = _cross_layer_closure(G_email, G_proximity, node)

        # Temporal betweenness features
        tb_feats = _tb_features_for_node(node, tb_matrix)
        feat.update(tb_feats)

        # Label
        feat["label"] = labels.get(node, 0)

        records.append(feat)

    df = pd.DataFrame(records)

    # Fill NaN in numeric columns with 0
    num_cols = [c for c in df.columns if c not in ("node",)]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


# ---------------------------------------------------------------------------
# Direct run — builds features.csv
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 3 — Feature Engineering")
    print("=" * 55)

    from src.data_loader import get_email_graph, get_proximity_graph, get_monthly_snapshots
    from src.analysis.temporal_betweenness import (
        temporal_betweenness_per_snapshot,
        build_tb_matrix,
    )

    CLEANED   = _ROOT / "data" / "cleaned"
    EMAIL_CSV = CLEANED / "email_edges_cleaned.csv"

    print("\nLoading graphs …")
    G_email = get_email_graph()
    G_prox  = get_proximity_graph()

    print("\nBuilding monthly snapshots …")
    snaps = get_monthly_snapshots()

    print(f"\nComputing temporal betweenness over {len(snaps)} snapshots …")
    tb_df = temporal_betweenness_per_snapshot(snaps)
    tb_mat = build_tb_matrix(tb_df)
    print(f"  TB matrix: {tb_mat.shape}")

    print("\nLoading email edges for label construction …")
    email_df = pd.read_csv(EMAIL_CSV)

    print("\nBuilding feature matrix …")
    feat_df = build_feature_matrix(G_email, G_prox, tb_mat, email_df)

    out = _ROOT / "data" / "features.csv"
    feat_df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({feat_df.shape[0]} rows, {feat_df.shape[1]} columns)")
    print(f"  At-risk: {(feat_df['label'] == 1).sum()}  |  Stable: {(feat_df['label'] == 0).sum()}")
    print("\nFeature columns:", [c for c in feat_df.columns if c != "node"])

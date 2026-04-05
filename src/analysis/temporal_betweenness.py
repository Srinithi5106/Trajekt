"""
src/analysis/temporal_betweenness.py
=====================================
Temporal Betweenness Centrality

Uses nx.betweenness_centrality per monthly snapshot (weighted, normalized).
Much faster than the custom BFS and consistent with the feature engineering
module.

Public API
----------
temporal_betweenness_per_snapshot(snapshots) -> pd.DataFrame
    columns: node | period | betweenness

build_tb_matrix(tb_df) -> pd.DataFrame
    nodes × periods pivot table for heatmap use
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def temporal_betweenness_per_snapshot(
    snapshots: dict[str, nx.Graph],
    weight_attr: str = "weight",
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Compute betweenness centrality for each node in each monthly snapshot.

    Parameters
    ----------
    snapshots : dict[str, nx.Graph]
        Dict keyed by period string ('YYYY-MM'), values are undirected graphs.
    weight_attr : str
        Edge attribute to use as weight.  If missing, falls back to unweighted.
    normalized : bool
        Whether to normalise betweenness scores (default True).

    Returns
    -------
    pd.DataFrame
        Columns: node | period | betweenness
    """
    records: list[dict] = []

    for period, G in sorted(snapshots.items()):
        if G.number_of_nodes() < 2:
            continue

        # Check whether weight attribute exists on edges
        has_weight = any(
            weight_attr in data for _, _, data in G.edges(data=True)
        )
        w = weight_attr if has_weight else None

        try:
            bc = nx.betweenness_centrality(G, normalized=normalized, weight=w)
        except Exception as e:
            print(f"  WARNING: betweenness failed for {period}: {e} — using unweighted")
            try:
                bc = nx.betweenness_centrality(G, normalized=normalized)
            except Exception:
                bc = {}

        for node, score in bc.items():
            records.append({"node": node, "period": period, "betweenness": float(score)})

    if not records:
        return pd.DataFrame(columns=["node", "period", "betweenness"])

    return pd.DataFrame(records)


def build_tb_matrix(tb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the temporal betweenness DataFrame into a nodes × periods matrix.

    Parameters
    ----------
    tb_df : pd.DataFrame
        Output of :func:`temporal_betweenness_per_snapshot`.
        Columns: node | period | betweenness

    Returns
    -------
    pd.DataFrame
        Index = node, columns = sorted period strings, values = betweenness.
        Missing values filled with 0.
    """
    if tb_df.empty:
        return pd.DataFrame()

    matrix = tb_df.pivot_table(
        index="node",
        columns="period",
        values="betweenness",
        aggfunc="mean",
    ).fillna(0)

    # Ensure columns are sorted chronologically
    matrix = matrix[sorted(matrix.columns)]
    return matrix
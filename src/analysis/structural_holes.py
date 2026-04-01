"""
structural_holes.py
===================
Stage 4 — Burt's Structural Constraint & Correlation Analysis
--------------------------------------------------------------
Identifies nodes that bridge otherwise disconnected clusters
(low constraint = structural hole spanners / brokers).

Uses NetworkX's built-in ``nx.constraint`` which implements Burt's
network constraint measure:

    C_i = Σ_j (p_ij + Σ_q p_iq * p_qj)²

where p_ij = w_ij / Σ_k w_ik  (proportion of i's network investment in j).

Low constraint → node bridges structural holes (broker position).
High constraint → node is embedded in a dense, redundant cluster.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import stats


def burt_constraint(
    G: nx.Graph,
    weight_attr: str = "weight",
) -> dict:
    """
    Compute Burt's network constraint for every node in *G*.

    Parameters
    ----------
    G : nx.Graph
        Undirected (or directed) graph with optional edge weights.
    weight_attr : str
        Name of the edge attribute to use as weight.
        If the attribute is missing on edges, NetworkX falls back to
        unweighted computation.

    Returns
    -------
    dict[node, float]
        Mapping of node → constraint value.
        Isolated nodes will have ``NaN`` constraint.
    """
    # nx.constraint returns {node: constraint} for all nodes
    # It handles weighted edges via the `weight` parameter.
    constraint_dict = nx.constraint(G, weight=weight_attr)

    # Convert None values (isolated nodes) to NaN for consistency
    return {
        node: (float("nan") if val is None else val)
        for node, val in constraint_dict.items()
    }


def homophily_constraint_correlation(
    h_dict: dict,
    c_dict: dict,
) -> tuple[float, float]:
    """
    Compute the Spearman rank correlation between per-node homophily
    and per-node constraint values.

    Parameters
    ----------
    h_dict : dict[node, float]
        Per-node Coleman homophily values.
    c_dict : dict[node, float]
        Per-node Burt constraint values.

    Returns
    -------
    tuple[float, float]
        (spearman_rho, p_value).
        Returns (NaN, NaN) if fewer than 3 valid node pairs exist.
    """
    # Align on common nodes
    common_nodes = set(h_dict.keys()) & set(c_dict.keys())

    h_vals = []
    c_vals = []
    for node in common_nodes:
        h = h_dict[node]
        c = c_dict[node]
        # Drop NaNs from both sides
        if np.isfinite(h) and np.isfinite(c):
            h_vals.append(h)
            c_vals.append(c)

    if len(h_vals) < 3:
        # Spearman requires at least 3 data points for a meaningful result
        return (float("nan"), float("nan"))

    rho, pvalue = stats.spearmanr(h_vals, c_vals)
    return (rho, pvalue)

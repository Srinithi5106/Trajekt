"""
homophily.py
============
Stage 4 — Coleman Homophily Index
---------------------------------
Measures the tendency of nodes to connect with same-department neighbours,
controlling for random baseline expectation.

Coleman's homophily index per node:
    w_same  = fraction of node's total edge weight going to same-dept neighbours
    p_same  = fraction of all nodes in the network that share this node's dept
    h       = (w_same - p_same) / (1 - p_same)

h = 0  → connections match random expectation
h = 1  → perfectly homophilous (all weight to same-dept)
h < 0  → heterophilous (prefers cross-dept connections)
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


def coleman_homophily(
    G: nx.Graph,
    dept_attr: str = "dept",
    weight_attr: str = "weight",
) -> dict:
    """
    Compute the Coleman homophily index for every node in *G*.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Graph whose nodes carry a *dept_attr* attribute and whose edges
        carry a *weight_attr* attribute.
    dept_attr : str
        Name of the node attribute storing department membership.
    weight_attr : str
        Name of the edge attribute storing numeric weight.
        Falls back to unweighted (weight = 1) if the attribute is missing.

    Returns
    -------
    dict[node, float]
        Mapping of node → Coleman h value.
        Nodes with no edges (isolated) receive ``float('nan')``.
    """
    N = G.number_of_nodes()
    if N == 0:
        return {}

    # --- department census (global baseline) ---------------------------------
    dept_counts: Counter = Counter()
    for _, d in G.nodes(data=dept_attr):
        dept_counts[d] += 1

    # p_same for each department = count_of_dept / N
    p_same_by_dept: dict[str, float] = {
        dept: count / N for dept, count in dept_counts.items()
    }

    # --- per-node homophily --------------------------------------------------
    h_dict: dict = {}

    for node in G.nodes():
        node_dept = G.nodes[node].get(dept_attr)

        # Collect neighbour weights
        total_weight = 0.0
        same_weight = 0.0

        for nbr in G.neighbors(node):
            w = G[node][nbr].get(weight_attr, 1.0)
            total_weight += w
            if G.nodes[nbr].get(dept_attr) == node_dept:
                same_weight += w

        if total_weight == 0:
            # Isolated node or all edges have zero weight
            h_dict[node] = float("nan")
            continue

        w_same = same_weight / total_weight
        p_same = p_same_by_dept.get(node_dept, 0.0)

        if p_same >= 1.0:
            # Every node belongs to the same dept → h is undefined
            h_dict[node] = float("nan")
        else:
            h_dict[node] = (w_same - p_same) / (1.0 - p_same)

    return h_dict


def aggregate_by_dept(
    node_h: dict,
    G: nx.Graph,
    dept_attr: str = "dept",
) -> pd.Series:
    """
    Compute the mean Coleman homophily index per department.

    Parameters
    ----------
    node_h : dict[node, float]
        Per-node homophily values (from :func:`coleman_homophily`).
    G : nx.Graph
        The graph used to look up each node's department.
    dept_attr : str
        Name of the node attribute storing department membership.

    Returns
    -------
    pd.Series
        Index = department name, values = mean h for that department.
        NaN values are excluded from the mean.
    """
    records = []
    for node, h_val in node_h.items():
        dept = G.nodes[node].get(dept_attr, "unknown")
        if not math.isnan(h_val):
            records.append({"dept": dept, "h": h_val})

    if not records:
        return pd.Series(dtype=float, name="mean_h")

    df = pd.DataFrame(records)
    return df.groupby("dept")["h"].mean().rename("mean_h")

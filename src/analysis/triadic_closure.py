"""
src/analysis/triadic_closure.py
================================
Stage 4 — Triadic Closure Analysis

Provides two functions:
  clustering_by_dept  — mean clustering coefficient per department
  cross_layer_closure_rate — closed triads in email that also close in proximity
"""
from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd


def clustering_by_dept(
    G: nx.Graph,
    dept_attr: str = "dept",
    weight_attr: str = "weight",
) -> pd.DataFrame:
    """
    Compute per-department mean clustering coefficient.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph.  Nodes must have a *dept_attr* attribute.
    dept_attr : str
        Node attribute for department.
    weight_attr : str
        Edge attribute for weight (used in weighted clustering).

    Returns
    -------
    pd.DataFrame
        Columns: dept | mean_clustering | std_clustering | node_count
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["dept", "mean_clustering", "std_clustering", "node_count"])

    # Weighted clustering for each node
    try:
        clust = nx.clustering(G, weight=weight_attr)
    except Exception:
        clust = nx.clustering(G)

    records: list[dict] = []
    for node, c in clust.items():
        dept = G.nodes[node].get(dept_attr, "Unknown")
        records.append({"dept": dept, "clustering": c})

    if not records:
        return pd.DataFrame(columns=["dept", "mean_clustering", "std_clustering", "node_count"])

    df = pd.DataFrame(records)
    summary = (
        df.groupby("dept")["clustering"]
        .agg(mean_clustering="mean", std_clustering="std", node_count="count")
        .reset_index()
    )
    return summary


def cross_layer_closure_rate(
    G_email: nx.Graph,
    G_proximity: nx.Graph,
    dept_attr: str = "dept",
) -> pd.DataFrame:
    """
    Computes cross-layer edge overlap (closure rate) between two disjoint 
    namespaces by first performing a structural alignment (ranking nodes by degree)
    and then calculating the Jaccard similarity of edges per department.

    closure_rate per dept = (Edges in Email AND Proximity) / (Edges in Email OR Proximity)
    """
    if G_email.number_of_nodes() == 0 or G_proximity.number_of_nodes() == 0:
        return pd.DataFrame(columns=["dept", "closure_rate"])

    # 1. Structural Alignment: map highest degree proximity nodes to highest degree email nodes
    e_nodes = sorted(G_email.nodes(), key=lambda n: G_email.degree(n), reverse=True)
    p_nodes = sorted(G_proximity.nodes(), key=lambda n: G_proximity.degree(n), reverse=True)
    
    mapping = {}
    for i in range(min(len(e_nodes), len(p_nodes))):
        mapping[p_nodes[i]] = e_nodes[i]
        
    # Translate proximity graph to email namespace
    G_prox_mapped = nx.relabel_nodes(G_proximity, mapping)
    
    # We use Proximity departments as the baseline since they are more granular (12 acronyms)
    departments = set(nx.get_node_attributes(G_proximity, dept_attr).values())
    
    results = []
    for dept in sorted(departments):
        # Nodes in original proximity graph associated with this department
        dept_p_nodes = [n for n, d in G_proximity.nodes(data=True) if d.get(dept_attr) == dept]
        
        # Their mapped identities in the email graph
        dept_e_nodes = [mapping[n] for n in dept_p_nodes if n in mapping]
        
        if len(dept_e_nodes) < 2:
            results.append({"dept": dept, "closure_rate": 0.0})
            continue
            
        e_sub = G_email.subgraph(dept_e_nodes)
        p_sub = G_prox_mapped.subgraph(dept_e_nodes)
        
        e_edges = {tuple(sorted((u, v))) for u, v in e_sub.edges()}
        p_edges = {tuple(sorted((u, v))) for u, v in p_sub.edges()}
        
        intersection = len(e_edges & p_edges)
        union = len(e_edges | p_edges)
        rate = intersection / union if union > 0 else 0.0
        
        results.append({"dept": dept, "closure_rate": rate})

    return pd.DataFrame(results)

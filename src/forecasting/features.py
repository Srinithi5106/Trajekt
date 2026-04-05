"""
Feature engineering for career-outcome forecasting.

Produces a feature matrix indexed by ``(node, month)`` containing:

**Static (per-snapshot) features**
* ``degree``            – undirected degree in the email graph
* ``clustering``        – local clustering coefficient
* ``burt_constraint``   – Burt's constraint (structural-hole measure)
* ``homophily_email``   – fraction of email neighbours in the same dept
* ``homophily_prox``    – fraction of proximity neighbours in the same dept
* ``cross_closure``     – triadic-closure rate across the two layers

**Temporal features** (from temporal-betweenness series)
* ``tb_current``        – temporal betweenness in the current month
* ``tb_3m_avg``         – rolling 3-month average of TB
* ``tb_trend``          – slope of TB over the last 3 months
"""

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

def _degree(G: nx.Graph, node):
    return G.degree(node) if G.has_node(node) else 0


def _clustering(G: nx.Graph, node):
    if not G.has_node(node):
        return 0.0
    return nx.clustering(G.to_undirected() if G.is_directed() else G, node)


def _burt_constraint(G: nx.Graph, node):
    """
    Burt's constraint for *node*. Uses NetworkX's built-in implementation
    on the undirected projection.
    """
    U = G.to_undirected() if G.is_directed() else G
    if not U.has_node(node) or U.degree(node) == 0:
        return 1.0                       # isolated → maximum constraint
    try:
        c = nx.constraint(U, [node])
        return c.get(node, 1.0)
    except Exception:
        return 1.0


def _homophily(G: nx.Graph, node, dept_map: dict):
    """Fraction of node's neighbours that share the same department."""
    U = G.to_undirected() if G.is_directed() else G
    if not U.has_node(node) or U.degree(node) == 0:
        return 0.0
    own_dept = dept_map.get(node, "unknown")
    if own_dept == "unknown":
        return 0.0
    neighbours = list(U.neighbors(node))
    same = sum(1 for n in neighbours if dept_map.get(n, "unknown") == own_dept)
    return same / len(neighbours)


def _cross_closure(G_email: nx.Graph, G_prox: nx.Graph, node):
    """
    Cross-layer triadic closure: fraction of open triads in the email
    layer that are closed in the proximity layer.
    """
    Ue = G_email.to_undirected() if G_email.is_directed() else G_email
    Up = G_prox.to_undirected() if G_prox.is_directed() else G_prox

    if not Ue.has_node(node):
        return 0.0

    nbrs = set(Ue.neighbors(node))
    if len(nbrs) < 2:
        return 0.0

    open_triads = 0
    closed_in_prox = 0
    nbrs_list = list(nbrs)

    for i in range(len(nbrs_list)):
        for j in range(i + 1, len(nbrs_list)):
            u, v = nbrs_list[i], nbrs_list[j]
            if not Ue.has_edge(u, v):          # open triad in email
                open_triads += 1
                if Up.has_node(u) and Up.has_node(v) and Up.has_edge(u, v):
                    closed_in_prox += 1

    return closed_in_prox / open_triads if open_triads > 0 else 0.0


# ---------------------------------------------------------------------------
# Temporal-betweenness helpers
# ---------------------------------------------------------------------------

def _tb_features(tb_series: pd.DataFrame, node, month_idx: int, months):
    """
    Derive TB current, 3-month avg, and trend for *node* at *month_idx*.
    ``tb_series`` is a DataFrame with nodes as index and month columns.
    """
    if node not in tb_series.index:
        return 0.0, 0.0, 0.0

    current = tb_series.loc[node, months[month_idx]] if month_idx < len(months) else 0.0

    window_start = max(0, month_idx - 2)
    window_vals = [
        tb_series.loc[node, months[k]]
        for k in range(window_start, month_idx + 1)
        if months[k] in tb_series.columns
    ]
    avg_3m = float(np.mean(window_vals)) if window_vals else 0.0

    # trend = slope of the window (simple linear fit)
    if len(window_vals) >= 2:
        x = np.arange(len(window_vals), dtype=float)
        slope = np.polyfit(x, window_vals, 1)[0]
    else:
        slope = 0.0

    return float(current), avg_3m, float(slope)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(
    df_email: pd.DataFrame,
    df_proximity: pd.DataFrame,
    df_departments: pd.DataFrame,
    tb_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix indexed by (node, month).

    Parameters
    ----------
    df_email : DataFrame
        Columns: sender, recipient, timestamp, department.
    df_proximity : DataFrame
        Columns: timestamp, i, j, duration.
    df_departments : DataFrame
        Columns: node_id, department.
    tb_series : DataFrame, optional
        Index = nodes, columns = month periods. If ``None``, temporal
        features are filled with zeros.

    Returns
    -------
    DataFrame with feature columns and ``(node, month)`` as index.
    """
    # ── parse timestamps and add month column ──────────────────────
    df_email = df_email.copy()
    df_email["timestamp"] = pd.to_datetime(df_email["timestamp"], utc=True)
    df_email["month"] = df_email["timestamp"].dt.to_period("M")

    months = sorted(df_email["month"].unique())

    # ── department map ─────────────────────────────────────────────
    dept_map: dict = {}
    if df_departments is not None and len(df_departments) > 0:
        dept_map = dict(zip(df_departments["node_id"].astype(str),
                            df_departments["department"]))
    # Also merge from email department column
    for _, row in df_email.drop_duplicates("sender").iterrows():
        if row["department"] != "unknown":
            dept_map[row["sender"]] = row["department"]

    # ── build proximity graph (aggregated, undirected) ─────────────
    G_prox = nx.Graph()
    if df_proximity is not None and len(df_proximity) > 0:
        for _, row in df_proximity.iterrows():
            u, v = str(row["i"]), str(row["j"])
            if G_prox.has_edge(u, v):
                G_prox[u][v]["weight"] += 1
            else:
                G_prox.add_edge(u, v, weight=1)

    # ── iterate over monthly snapshots ─────────────────────────────
    records = []
    for idx, month in enumerate(months):
        snap = df_email[df_email["month"] == month]

        # build directed email graph for this month
        G_email = nx.DiGraph()
        for _, row in snap.iterrows():
            s, r = row["sender"], row["recipient"]
            if G_email.has_edge(s, r):
                G_email[s][r]["weight"] += 1
            else:
                G_email.add_edge(s, r, weight=1)

        nodes_in_snap = set(G_email.nodes())

        for node in nodes_in_snap:
            feat = {
                "node": node,
                "month": month,
                "degree": _degree(G_email, node),
                "clustering": _clustering(G_email, node),
                "burt_constraint": _burt_constraint(G_email, node),
                "homophily_email": _homophily(G_email, node, dept_map),
                "homophily_prox": _homophily(G_prox, node, dept_map),
                "cross_closure": _cross_closure(G_email, G_prox, node),
            }

            # temporal betweenness features
            if tb_series is not None and len(tb_series.columns) > 0:
                tb_cur, tb_avg, tb_trend = _tb_features(
                    tb_series, node, idx, months
                )
            else:
                tb_cur = tb_avg = tb_trend = 0.0

            feat["tb_current"] = tb_cur
            feat["tb_3m_avg"] = tb_avg
            feat["tb_trend"] = tb_trend

            records.append(feat)

    features_df = pd.DataFrame(records)
    return features_df

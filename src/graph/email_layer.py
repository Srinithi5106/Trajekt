"""
graph_email_layer.py
====================
Phase 02 — Email Layer (E₁) Construction
-----------------------------------------
Builds a directed, weighted email graph from parsed Enron data, then
collapses it to an undirected graph for triadic-closure analysis.
Also slices the graph into monthly temporal snapshots.

Expected input columns (from Phase 01 parsing):
    sender      : str   — email address
    receiver    : str   — email address (one row per recipient)
    timestamp   : datetime64[ns, UTC]
    sender_dept : str   — inferred from folder path
    recv_dept   : str   — inferred from folder path (may be NaN)
"""

import pandas as pd
import networkx as nx
from collections import defaultdict


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_edge_weighted(G: nx.DiGraph, u: str, v: str, weight: float = 1.0) -> None:
    """Increment edge weight if edge exists, else create it."""
    if G.has_edge(u, v):
        G[u][v]["weight"] += weight
    else:
        G.add_edge(u, v, weight=weight)


# ── main builder ─────────────────────────────────────────────────────────────

def build_email_graph(
    df: pd.DataFrame,
    dept_map: dict[str, str] | None = None,
) -> nx.DiGraph:
    """
    Build a directed, weighted email graph from the parsed Enron DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: sender, receiver, timestamp, sender_dept.
    dept_map : dict[str, str] | None
        Optional override mapping address → dept (e.g. from the
        Enron↔SocioPatterns alignment table built in Phase 01).

    Returns
    -------
    nx.DiGraph
        Directed graph.  Edge attributes:
            weight      — number of emails sent u→v
            first_ts    — earliest timestamp of the edge (pd.Timestamp)
            last_ts     — latest  timestamp of the edge (pd.Timestamp)
        Node attributes:
            dept        — department string
            email_count — total emails sent by this node
    """
    required = {"sender", "receiver", "timestamp", "sender_dept"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    G = nx.DiGraph(name="Enron_Email_E1")

    # track per-edge timestamps without storing full lists
    edge_first: dict[tuple, pd.Timestamp] = {}
    edge_last:  dict[tuple, pd.Timestamp] = {}
    node_send_count: dict[str, int] = defaultdict(int)

    for row in df.itertuples(index=False):
        u, v, ts, dept = (
            row.sender,
            row.receiver,
            row.timestamp,
            row.sender_dept,
        )

        # add nodes with dept attribute
        for node, d in ((u, dept), (v, getattr(row, "recv_dept", None))):
            if node not in G:
                # prefer dept_map override if provided
                resolved_dept = (
                    dept_map.get(node, d) if dept_map else d
                )
                G.add_node(node, dept=resolved_dept, email_count=0)
            elif dept_map and node in dept_map:
                G.nodes[node]["dept"] = dept_map[node]

        _add_edge_weighted(G, u, v)
        node_send_count[u] += 1

        key = (u, v)
        if key not in edge_first or ts < edge_first[key]:
            edge_first[key] = ts
        if key not in edge_last or ts > edge_last[key]:
            edge_last[key] = ts

    # attach timestamp metadata to edges
    for (u, v), ts_first in edge_first.items():
        G[u][v]["first_ts"] = ts_first
        G[u][v]["last_ts"]  = edge_last[(u, v)]

    # attach send counts to nodes
    for node, cnt in node_send_count.items():
        G.nodes[node]["email_count"] = cnt

    return G


def collapse_to_undirected(G_dir: nx.DiGraph) -> nx.Graph:
    """
    Collapse directed email graph → undirected for triadic-closure analysis.

    Edge weight = sum of weights in both directions (u→v) + (v→u).
    """
    G_un = nx.Graph(name="Enron_Email_E1_undirected")

    for node, attrs in G_dir.nodes(data=True):
        G_un.add_node(node, **attrs)

    for u, v, data in G_dir.edges(data=True):
        w = data.get("weight", 1.0)
        if G_un.has_edge(u, v):
            G_un[u][v]["weight"] += w
        else:
            G_un.add_edge(u, v, weight=w)

    return G_un


# ── temporal snapshots ────────────────────────────────────────────────────────

def build_monthly_snapshots(
    df: pd.DataFrame,
    dept_map: dict[str, str] | None = None,
) -> dict[str, nx.DiGraph]:
    """
    Slice the email DataFrame into monthly periods and build one directed
    graph per month.

    Returns
    -------
    dict[str, nx.DiGraph]
        Keys are period strings like '1999-11', '2000-01', …
        Values are directed graphs for that month.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["period"] = df["timestamp"].dt.to_period("M")

    snapshots: dict[str, nx.DiGraph] = {}
    for period, group in df.groupby("period"):
        key = str(period)
        snapshots[key] = build_email_graph(group, dept_map=dept_map)

    return snapshots


# ── quick sanity helper ───────────────────────────────────────────────────────

def email_graph_summary(G: nx.DiGraph | nx.Graph) -> dict:
    """Return a dict of basic stats for logging / unit-test assertions."""
    is_directed = G.is_directed()
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": is_directed,
        "avg_weight": (
            sum(d["weight"] for _, _, d in G.edges(data=True)) / max(G.number_of_edges(), 1)
        ),
        "depts": list({d for _, d in G.nodes(data="dept") if d}),
        "isolated_nodes": len(list(nx.isolates(G))),
        "density": nx.density(G),
    }


# ── demo / manual smoke-test ──────────────────────────────────────────────────

if __name__ == "__main__":
    # build tiny synthetic frame to verify logic without real data
    demo_df = pd.DataFrame(
        {
            "sender":      ["alice@enron.com", "alice@enron.com", "bob@enron.com", "carol@enron.com"],
            "receiver":    ["bob@enron.com",   "carol@enron.com", "carol@enron.com", "alice@enron.com"],
            "timestamp":   pd.to_datetime(
                               ["2001-01-05", "2001-01-07", "2001-02-03", "2001-02-10"]
                           ).tz_localize("UTC"),
            "sender_dept": ["trading",         "trading",         "legal",           "finance"],
            "recv_dept":   ["legal",            "finance",         "finance",         "trading"],
        }
    )

    G_dir = build_email_graph(demo_df)
    print("Directed graph summary:", email_graph_summary(G_dir))

    G_un = collapse_to_undirected(G_dir)
    print("Undirected graph summary:", email_graph_summary(G_un))

    snaps = build_monthly_snapshots(demo_df)
    print("Monthly snapshots:", list(snaps.keys()))
    for period, snap in snaps.items():
        print(f"  {period}: {snap.number_of_nodes()} nodes, {snap.number_of_edges()} edges")

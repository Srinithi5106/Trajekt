"""
graph_proximity_layer.py
========================
Phase 02 — Proximity Layer (E₂) Construction
---------------------------------------------
Builds an undirected, weighted proximity graph from SocioPatterns InVS13 data.

SocioPatterns tij format (space-separated, no header):
    t   — UNIX timestamp (int, 20-second resolution)
    i   — node id (int)
    j   — node id (int)

Each row means nodes i and j were in face-to-face contact during the 20s
window starting at t.  Weight = total contact seconds (rows × 20).

Minimum threshold: pairs with total contact < 60 s are dropped (checklist spec).
"""

import pandas as pd
import networkx as nx


# ── loaders ───────────────────────────────────────────────────────────────────

def load_tij(filepath: str) -> pd.DataFrame:
    """
    Load a SocioPatterns tij_*.dat file.

    Returns
    -------
    pd.DataFrame with columns: t (int), i (int), j (int)
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["t", "i", "j"],
        dtype={"t": int, "i": int, "j": int},
        comment="#",
    )
    return df


def load_metadata(filepath: str) -> pd.DataFrame:
    """
    Load SocioPatterns metadata_*.txt.

    Expected columns: id, dept  (tab or space separated, may have header).
    Returns a DataFrame with columns: node_id (int), dept (str).
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        comment="#",
    )

    # handle files that include a text header row
    if df.iloc[0, 0].astype(str).isdigit() is False:
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = ["node_id", "dept"] + list(df.columns[2:])
    df["node_id"] = df["node_id"].astype(int)
    return df[["node_id", "dept"]]


# ── main builder ──────────────────────────────────────────────────────────────

CONTACT_WINDOW_SECONDS = 20   # SocioPatterns resolution
MIN_CONTACT_SECONDS    = 60   # checklist threshold (≥ 3 windows)


def build_proximity_graph(
    tij_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    min_contact_seconds: int = MIN_CONTACT_SECONDS,
) -> nx.Graph:
    """
    Build an undirected, weighted proximity graph.

    Parameters
    ----------
    tij_df : pd.DataFrame
        Columns t, i, j (from load_tij).
    metadata_df : pd.DataFrame | None
        Columns node_id, dept (from load_metadata).  If None, dept = 'unknown'.
    min_contact_seconds : int
        Minimum cumulative contact seconds required to keep an edge.

    Returns
    -------
    nx.Graph
        Edge attributes:
            weight          — total contact seconds
            contact_windows — number of 20-second windows
        Node attributes:
            dept            — department string
    """
    # ── aggregate contact duration per pair ──────────────────────────────────
    # normalise so lower id is always i (undirected)
    df = tij_df.copy()
    df["i"], df["j"] = (
        df[["i", "j"]].min(axis=1),
        df[["i", "j"]].max(axis=1),
    )

    agg = (
        df.groupby(["i", "j"])
        .size()
        .reset_index(name="contact_windows")
    )
    agg["weight"] = agg["contact_windows"] * CONTACT_WINDOW_SECONDS

    # ── apply minimum contact threshold ──────────────────────────────────────
    agg = agg[agg["weight"] >= min_contact_seconds].copy()

    # ── dept lookup ──────────────────────────────────────────────────────────
    dept_lookup: dict[int, str] = {}
    if metadata_df is not None:
        dept_lookup = dict(
            zip(metadata_df["node_id"], metadata_df["dept"])
        )

    # ── build graph ──────────────────────────────────────────────────────────
    G = nx.Graph(name="SocioPatterns_Proximity_E2")

    all_nodes = set(agg["i"]) | set(agg["j"])
    for node in all_nodes:
        G.add_node(node, dept=dept_lookup.get(node, "unknown"))

    for row in agg.itertuples(index=False):
        G.add_edge(
            row.i,
            row.j,
            weight=row.weight,
            contact_windows=row.contact_windows,
        )

    return G


# ── utility ───────────────────────────────────────────────────────────────────

def proximity_graph_summary(G: nx.Graph) -> dict:
    """Return basic stats dict for logging / assertions."""
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": G.is_directed(),
        "min_weight_s":  min(weights, default=0),
        "max_weight_s":  max(weights, default=0),
        "mean_weight_s": sum(weights) / max(len(weights), 1),
        "depts": list({d for _, d in G.nodes(data="dept") if d != "unknown"}),
        "density": nx.density(G),
    }


def filter_by_dept(G: nx.Graph, dept: str) -> nx.Graph:
    """Return a subgraph induced on nodes belonging to a specific dept."""
    nodes = [n for n, d in G.nodes(data="dept") if d == dept]
    return G.subgraph(nodes).copy()


# ── demo / manual smoke-test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import io

    # minimal synthetic tij data (each row = 20s contact window)
    tij_raw = """
1351756800 1 2
1351756800 1 3
1351756820 1 2
1351756820 2 3
1351756840 1 2
1351756840 1 3
1351756860 3 4
"""
    # node 1↔2: 3 windows = 60s  ← exactly at threshold → keep
    # node 1↔3: 2 windows = 40s  ← below threshold → drop
    # node 2↔3: 1 window  = 20s  ← below threshold → drop
    # node 3↔4: 1 window  = 20s  ← below threshold → drop

    meta_raw = """
1 Dept_A
2 Dept_A
3 Dept_B
4 Dept_B
"""
    tij_df  = load_tij(io.StringIO(tij_raw.strip()))
    meta_df = load_metadata(io.StringIO(meta_raw.strip()))

    G = build_proximity_graph(tij_df, meta_df, min_contact_seconds=60)

    print("Proximity graph summary:", proximity_graph_summary(G))
    print("Edges:")
    for u, v, d in G.edges(data=True):
        print(f"  {u}↔{v}  weight={d['weight']}s  windows={d['contact_windows']}")
    print("Node depts:")
    for n, d in G.nodes(data=True):
        print(f"  node {n}: dept={d['dept']}")

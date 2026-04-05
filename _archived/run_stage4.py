"""
run_stage4.py
=============
Stage 4 Runner -- Homophily + Structural Holes Analysis
-------------------------------------------------------
Loads processed CSV data, builds weighted NetworkX graphs for both
the email and proximity layers, computes Coleman homophily and Burt
constraint metrics, and outputs a cross-layer comparison table.

The email layer uses email_edges_sampled.csv (top 200 most active users)
rather than the full 76K-node graph, because Burt's constraint is O(n*d^2)
and infeasible on the full graph without hours of compute.

Usage:
    python -m src.analysis.run_stage4
    # or
    python src/analysis/run_stage4.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows to avoid cp1252 encoding crashes
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import src.analysis.*
# regardless of how this script is invoked.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # src/analysis/
_PROJECT_ROOT = _THIS_DIR.parent.parent              # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.homophily import aggregate_by_dept, coleman_homophily
from src.analysis.structural_holes import burt_constraint, homophily_constraint_correlation

# ---------------------------------------------------------------------------
# Configuration -- paths relative to project root
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"

EMAIL_SAMPLED_FILE = DATA_DIR / "email_edges_sampled.csv"
EMAIL_EDGES_FILE = DATA_DIR / "email_edges.csv"
PROXIMITY_FILE = DATA_DIR / "proximity_edges.csv"
NODE_DEPTS_FILE = DATA_DIR / "node_departments.csv"

OUTPUT_FILE = DATA_DIR / "stage4_results.csv"


# ===================================================================
# Graph construction helpers
# ===================================================================

def build_email_graph(sampled_path: Path) -> nx.Graph:
    """
    Build an undirected, weighted graph from the sampled email edges CSV.

    The sampled CSV contains the top 200 most active users (produced by
    the ingestion pipeline). We aggregate sender-recipient pairs into
    edge weights (number of emails between a pair).

    Parameters
    ----------
    sampled_path : Path
        Path to ``email_edges_sampled.csv``.

    Returns
    -------
    nx.Graph
        Undirected graph with ``weight`` edge attribute.
    """
    df = pd.read_csv(sampled_path)
    cols = df.columns.tolist()
    print(f"  [+] Loaded {sampled_path.name}: {df.shape[0]:,} rows, columns={cols}")

    # Detect column names dynamically
    if "sender" in cols:
        src_col = "sender"
    else:
        src_col = cols[0]
        print(f"  [!] 'sender' column not found; using '{src_col}'")

    if "recipient" in cols:
        dst_col = "recipient"
    else:
        dst_col = cols[1]
        print(f"  [!] 'recipient' column not found; using '{dst_col}'")

    # Check for pre-aggregated weight column
    if "weight" in cols:
        wt_col = "weight"
    else:
        # Aggregate: count rows per sender-recipient pair
        wt_col = None

    if wt_col:
        agg = df.groupby([src_col, dst_col])[wt_col].sum().reset_index()
    else:
        agg = df.groupby([src_col, dst_col]).size().reset_index(name="weight")
        wt_col = "weight"

    G = nx.Graph(name="Email_Layer")
    for row in agg.itertuples(index=False):
        u = getattr(row, src_col)
        v = getattr(row, dst_col)
        w = getattr(row, wt_col)

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=float(w))

    print(f"  [+] Email graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def build_proximity_graph(prox_path: Path) -> nx.Graph:
    """
    Build an undirected, weighted graph from the proximity edges CSV.

    Weight = sum of ``duration`` for each unique node pair.

    Parameters
    ----------
    prox_path : Path
        Path to ``proximity_edges.csv``.

    Returns
    -------
    nx.Graph
        Undirected graph with ``weight`` edge attribute (total contact seconds).
    """
    df = pd.read_csv(prox_path)
    cols = df.columns.tolist()
    print(f"  [+] Loaded {prox_path.name}: {df.shape[0]:,} rows, columns={cols}")

    # Detect node id columns and duration/weight column
    if "i" in cols and "j" in cols:
        id_col_a, id_col_b = "i", "j"
    else:
        id_col_a, id_col_b = cols[1], cols[2]
        print(f"  [!] Expected columns 'i','j' not found; using '{id_col_a}','{id_col_b}'")

    if "duration" in cols:
        dur_col = "duration"
    elif "weight" in cols:
        dur_col = "weight"
        print("  [!] 'duration' column not found; using 'weight' as duration proxy")
    else:
        dur_col = None
        print("  [!] No 'duration' or 'weight' column found; using unweighted (1)")

    # Aggregate total duration per pair
    if dur_col:
        agg = df.groupby([id_col_a, id_col_b])[dur_col].sum().reset_index()
        agg.rename(columns={dur_col: "weight"}, inplace=True)
    else:
        agg = df.groupby([id_col_a, id_col_b]).size().reset_index(name="weight")

    G = nx.Graph(name="Proximity_Layer")
    for row in agg.itertuples(index=False):
        i_node = getattr(row, id_col_a)
        j_node = getattr(row, id_col_b)
        G.add_edge(i_node, j_node, weight=float(row.weight))

    print(f"  [+] Proximity graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ===================================================================
# Department assignment helpers
# ===================================================================

def infer_email_dept_map(email_edges_path: Path) -> dict[str, str]:
    """
    Infer a sender -> department mapping from the full email edges CSV.

    Uses the most frequently occurring department per sender (mode).
    Senders whose only department is 'unknown' are still mapped to 'unknown'.

    Parameters
    ----------
    email_edges_path : Path
        Path to ``email_edges.csv``.

    Returns
    -------
    dict[str, str]
        Mapping of email address -> inferred department.
    """
    df = pd.read_csv(email_edges_path)
    cols = df.columns.tolist()
    print(f"  [+] Loaded {email_edges_path.name}: {df.shape[0]:,} rows, columns={cols}")

    # Detect columns dynamically
    if "sender" in cols:
        sender_col = "sender"
    else:
        sender_col = cols[0]
        print(f"  [!] 'sender' column not found; using '{sender_col}'")

    if "department" in cols:
        dept_col = "department"
    else:
        candidates = [c for c in cols if "dept" in c.lower()]
        if candidates:
            dept_col = candidates[0]
            print(f"  [!] 'department' column not found; using '{dept_col}'")
        else:
            print("  [!] No department-like column found in email_edges.csv; "
                  "all nodes will be 'unknown'")
            return {}

    null_count = df[dept_col].isnull().sum()
    if null_count > 0:
        print(f"  [!] {null_count:,} null values in '{dept_col}' column")

    # Most frequent department per sender
    dept_map = (
        df.groupby(sender_col)[dept_col]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .to_dict()
    )

    known = sum(1 for v in dept_map.values() if v != "unknown")
    print(f"  [+] Inferred dept for {len(dept_map):,} senders "
          f"({known:,} known, {len(dept_map) - known:,} unknown)")

    return dept_map


def assign_email_depts(G: nx.Graph, dept_map: dict[str, str]) -> None:
    """
    Set the 'dept' node attribute on the email graph using the inferred map.

    Nodes not found in the map are labelled 'unknown'.

    Parameters
    ----------
    G : nx.Graph
        The email graph.
    dept_map : dict[str, str]
        sender -> department mapping.
    """
    attrs = {node: dept_map.get(node, "unknown") for node in G.nodes()}
    nx.set_node_attributes(G, attrs, "dept")

    dept_counts = pd.Series(attrs).value_counts()
    print(f"  [+] Email graph dept distribution:")
    for dept, cnt in dept_counts.items():
        print(f"      {dept:>12s}: {cnt:,}")


def assign_proximity_depts(G: nx.Graph, node_dept_path: Path) -> None:
    """
    Set the 'dept' node attribute on the proximity graph from node_departments.csv.

    Parameters
    ----------
    G : nx.Graph
        The proximity graph.
    node_dept_path : Path
        Path to ``node_departments.csv``.
    """
    df = pd.read_csv(node_dept_path)
    cols = df.columns.tolist()
    print(f"  [+] Loaded {node_dept_path.name}: {df.shape[0]} rows, columns={cols}")

    if "node_id" in cols:
        id_col = "node_id"
    else:
        id_col = cols[0]
        print(f"  [!] 'node_id' column not found; using '{id_col}'")

    if "department" in cols:
        dept_col = "department"
    elif "dept" in cols:
        dept_col = "dept"
    else:
        dept_col = cols[1]
        print(f"  [!] No department-like column found; using '{dept_col}'")

    dept_lookup = dict(zip(df[id_col], df[dept_col]))

    attrs = {node: dept_lookup.get(node, "unknown") for node in G.nodes()}
    nx.set_node_attributes(G, attrs, "dept")

    assigned = sum(1 for v in attrs.values() if v != "unknown")
    print(f"  [+] Proximity graph: {assigned}/{G.number_of_nodes()} nodes matched to dept")


# ===================================================================
# Main pipeline
# ===================================================================

def main() -> None:
    """Run the full Stage 4 analysis pipeline."""
    t0 = time.time()

    print("=" * 65)
    print("  STAGE 4 -- Homophily + Structural Holes Analysis")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Check that data files exist
    # ------------------------------------------------------------------
    required_files = {
        "email_edges_sampled": EMAIL_SAMPLED_FILE,
        "email_edges": EMAIL_EDGES_FILE,
        "proximity_edges": PROXIMITY_FILE,
        "node_departments": NODE_DEPTS_FILE,
    }
    for label, path in required_files.items():
        if not path.exists():
            print(f"  [X] MISSING: {path}")
            sys.exit(1)
        print(f"  [OK] Found: {path.name}")

    # ------------------------------------------------------------------
    # 2. Build graphs
    # ------------------------------------------------------------------
    print("\n-- Building email graph (sampled, top 200 users) --")
    G_email = build_email_graph(EMAIL_SAMPLED_FILE)

    print("\n-- Building proximity graph --")
    G_proximity = build_proximity_graph(PROXIMITY_FILE)

    # ------------------------------------------------------------------
    # 3. Assign department attributes
    # ------------------------------------------------------------------
    print("\n-- Assigning email departments --")
    email_dept_map = infer_email_dept_map(EMAIL_EDGES_FILE)
    assign_email_depts(G_email, email_dept_map)

    print("\n-- Assigning proximity departments --")
    assign_proximity_depts(G_proximity, NODE_DEPTS_FILE)

    # ------------------------------------------------------------------
    # 4. Coleman homophily
    # ------------------------------------------------------------------
    print("\n-- Computing Coleman homophily --")
    t1 = time.time()
    h_email = coleman_homophily(G_email, dept_attr="dept", weight_attr="weight")
    h_proximity = coleman_homophily(G_proximity, dept_attr="dept", weight_attr="weight")
    print(f"  [+] Homophily computed in {time.time() - t1:.1f}s")

    finite_h_email = [v for v in h_email.values() if np.isfinite(v)]
    finite_h_prox = [v for v in h_proximity.values() if np.isfinite(v)]
    print(f"  [+] Email homophily:     mean={np.mean(finite_h_email):.4f}, "
          f"median={np.median(finite_h_email):.4f} ({len(finite_h_email):,} finite nodes)")
    print(f"  [+] Proximity homophily: mean={np.mean(finite_h_prox):.4f}, "
          f"median={np.median(finite_h_prox):.4f} ({len(finite_h_prox):,} finite nodes)")

    # ------------------------------------------------------------------
    # 5. Burt's constraint
    # ------------------------------------------------------------------
    print("\n-- Computing Burt's constraint --")
    t2 = time.time()
    c_email = burt_constraint(G_email, weight_attr="weight")
    print(f"  [+] Email constraint computed in {time.time() - t2:.1f}s")

    t3 = time.time()
    c_proximity = burt_constraint(G_proximity, weight_attr="weight")
    print(f"  [+] Proximity constraint computed in {time.time() - t3:.1f}s")

    finite_c_email = [v for v in c_email.values() if np.isfinite(v)]
    finite_c_prox = [v for v in c_proximity.values() if np.isfinite(v)]
    print(f"  [+] Email constraint:     mean={np.mean(finite_c_email):.4f}, "
          f"median={np.median(finite_c_email):.4f} ({len(finite_c_email):,} finite nodes)")
    print(f"  [+] Proximity constraint: mean={np.mean(finite_c_prox):.4f}, "
          f"median={np.median(finite_c_prox):.4f} ({len(finite_c_prox):,} finite nodes)")

    # ------------------------------------------------------------------
    # 6. Homophily <-> Constraint correlation (per layer)
    # ------------------------------------------------------------------
    print("\n-- Homophily-Constraint correlation --")
    rho_email, p_email = homophily_constraint_correlation(h_email, c_email)
    rho_prox, p_prox = homophily_constraint_correlation(h_proximity, c_proximity)

    print(f"  [+] Email layer:     Spearman rho = {rho_email:.4f}, p = {p_email:.2e}")
    print(f"  [+] Proximity layer: Spearman rho = {rho_prox:.4f}, p = {p_prox:.2e}")

    # ------------------------------------------------------------------
    # 7. Per-department summary table
    # ------------------------------------------------------------------
    print("\n-- Per-department summary --")

    mean_h_email = aggregate_by_dept(h_email, G_email, dept_attr="dept")
    mean_h_prox = aggregate_by_dept(h_proximity, G_proximity, dept_attr="dept")

    # Mean constraint per department
    def _mean_constraint_by_dept(c_dict, G, dept_attr="dept"):
        """Compute mean Burt constraint per department."""
        records = []
        for node, c_val in c_dict.items():
            dept = G.nodes[node].get(dept_attr, "unknown")
            if np.isfinite(c_val):
                records.append({"dept": dept, "constraint": c_val})
        if not records:
            return pd.Series(dtype=float, name="mean_constraint")
        df = pd.DataFrame(records)
        return df.groupby("dept")["constraint"].mean().rename("mean_constraint")

    mean_c_email = _mean_constraint_by_dept(c_email, G_email)
    mean_c_prox = _mean_constraint_by_dept(c_proximity, G_proximity)

    # Merge into a single table
    all_depts = sorted(
        set(mean_h_email.index)
        | set(mean_h_prox.index)
        | set(mean_c_email.index)
        | set(mean_c_prox.index)
    )

    summary_rows = []
    for dept in all_depts:
        summary_rows.append({
            "dept": dept,
            "mean_h_email": mean_h_email.get(dept, float("nan")),
            "mean_h_proximity": mean_h_prox.get(dept, float("nan")),
            "mean_constraint_email": mean_c_email.get(dept, float("nan")),
            "mean_constraint_proximity": mean_c_prox.get(dept, float("nan")),
        })

    results_df = pd.DataFrame(summary_rows).set_index("dept")

    print()
    print(results_df.to_string(float_format="{:.4f}".format))
    print()

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    results_df.to_csv(OUTPUT_FILE)
    print(f"  [OK] Results saved to {OUTPUT_FILE}")

    # Correlation summary
    print()
    print("=" * 65)
    print("  CORRELATION SUMMARY")
    print("=" * 65)
    print(f"  Email layer:     Spearman rho = {rho_email:+.4f}  (p = {p_email:.2e})")
    print(f"  Proximity layer: Spearman rho = {rho_prox:+.4f}  (p = {p_prox:.2e})")
    print("=" * 65)
    print(f"\n  Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

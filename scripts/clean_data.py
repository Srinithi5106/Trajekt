"""
scripts/clean_data.py
=====================
Phase 0.5 — Data Validation & Cleaning

Reads raw files from /data/, cleans them, and writes to /data/cleaned/.
All downstream phases read exclusively from /data/cleaned/.

Run once:
    python scripts/clean_data.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data"
DATA_CLEAN = ROOT / "data" / "cleaned"
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

def sep(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Email Edges — Cleaned
# ---------------------------------------------------------------------------
def clean_email_edges() -> pd.DataFrame:
    sep("Cleaning email_edges.csv")
    src = DATA_RAW / "email_edges.csv"
    if not src.exists():
        print(f"WARNING: {src.name} not found — cannot clean email edges.")
        return pd.DataFrame()

    print(f"  Loading {src} …")
    df = pd.read_csv(src, encoding="utf-8", encoding_errors="replace")
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns:   {list(df.columns)}")

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Normalise email addresses to lowercase
    for col in ["sender", "recipient"]:
        if col in df.columns:
            df[col] = df[col].str.lower()

    # Parse timestamps robustly
    print("  Parsing timestamps …")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = df["timestamp"].isnull().sum()
    if bad_ts:
        print(f"  WARNING: {bad_ts:,} unparseable timestamps — dropping.")
    df = df.dropna(subset=["timestamp"])

    # Filter to 1999-01-01 through 2002-12-31
    t_min = pd.Timestamp("1999-01-01", tz="UTC")
    t_max = pd.Timestamp("2002-12-31 23:59:59", tz="UTC")
    before = len(df)
    df = df[(df["timestamp"] >= t_min) & (df["timestamp"] <= t_max)]
    print(f"  Time filter: {before:,} → {len(df):,} rows")

    # Remove self-loops
    if "sender" in df.columns and "recipient" in df.columns:
        before = len(df)
        df = df[df["sender"] != df["recipient"]]
        print(f"  Self-loops removed: {before - len(df):,}")

    # Remove duplicate (sender, recipient, timestamp) rows
    before = len(df)
    df = df.drop_duplicates(subset=["sender", "recipient", "timestamp"])
    print(f"  Duplicates removed: {before - len(df):,}")

    # Keep only needed columns
    keep_cols = [c for c in ["sender", "recipient", "timestamp", "department"] if c in df.columns]
    df = df[keep_cols]

    # Fill missing department
    if "department" in df.columns:
        df["department"] = df["department"].fillna("unknown")

    out = DATA_CLEAN / "email_edges_cleaned.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out} ({df.shape[0]:,} rows)")
    return df


# ---------------------------------------------------------------------------
# 2. Email Edges Aggregated — Cleaned
# ---------------------------------------------------------------------------
def clean_email_aggregated(email_df: pd.DataFrame) -> pd.DataFrame:
    sep("Cleaning email_edges_aggregated.csv")

    # Re-derive from cleaned email data if available
    if not email_df.empty:
        print("  Deriving from cleaned email edges …")
        agg = (
            email_df.groupby(["sender", "recipient"])
            .size()
            .reset_index(name="weight")
        )
        print(f"  Derived aggregated shape: {agg.shape}")
    else:
        # Fallback: load original aggregated file
        src = DATA_RAW / "email_edges_aggregated.csv"
        if not src.exists():
            print(f"  WARNING: {src.name} not found and no cleaned email edges available.")
            return pd.DataFrame()
        agg = pd.read_csv(src)
        print(f"  Loaded fallback {src.name}: {agg.shape}")
        for col in agg.select_dtypes(include="object").columns:
            agg[col] = agg[col].str.strip().str.lower()

    # Remove self-loops (should be none, but guard)
    before = len(agg)
    agg = agg[agg["sender"] != agg["recipient"]]
    if before != len(agg):
        print(f"  Self-loops removed: {before - len(agg):,}")

    # Keep only top-200 most active nodes if node count > 200
    all_nodes = pd.concat([agg["sender"], agg["recipient"]])
    node_degrees = all_nodes.value_counts()
    n_nodes = len(node_degrees)
    print(f"  Unique nodes: {n_nodes:,}")

    if n_nodes > 200:
        top_200 = node_degrees.head(200).index
        before = len(agg)
        agg = agg[agg["sender"].isin(top_200) & agg["recipient"].isin(top_200)]
        print(f"  Filtered to top-200 nodes: {before:,} → {len(agg):,} edges")

    out = DATA_CLEAN / "email_edges_aggregated_cleaned.csv"
    agg.to_csv(out, index=False)
    print(f"  Saved → {out} ({agg.shape[0]:,} rows)")
    return agg


# ---------------------------------------------------------------------------
# 3. Proximity Edges — Cleaned
# ---------------------------------------------------------------------------
def clean_proximity_edges() -> pd.DataFrame:
    sep("Cleaning proximity_edges.csv")
    src = DATA_RAW / "proximity_edges.csv"
    if not src.exists():
        print(f"WARNING: {src.name} not found.")
        return pd.DataFrame()

    df = pd.read_csv(src)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns:   {list(df.columns)}")

    # Detect node columns
    if "i" in df.columns and "j" in df.columns:
        i_col, j_col = "i", "j"
    else:
        i_col, j_col = df.columns[1], df.columns[2]
        print(f"  WARNING: expected 'i','j'; using '{i_col}','{j_col}'")

    # Detect duration column
    if "duration" in df.columns:
        dur_col = "duration"
    elif "weight" in df.columns:
        dur_col = "weight"
    else:
        df["duration"] = 20  # each row = one 20s window
        dur_col = "duration"
        print("  WARNING: no duration column — assuming 20s per row")

    # Ensure directed edges are normalised (lower id first for undirected)
    df["i_norm"] = df[[i_col, j_col]].min(axis=1)
    df["j_norm"] = df[[i_col, j_col]].max(axis=1)

    # Remove self-loops
    before = len(df)
    df = df[df["i_norm"] != df["j_norm"]]
    print(f"  Self-loops removed: {before - len(df):,}")

    # Remove zero/negative duration
    before = len(df)
    df = df[df[dur_col] > 0]
    print(f"  Zero-duration rows removed: {before - len(df):,}")

    # Aggregate: sum duration per pair
    agg = (
        df.groupby(["i_norm", "j_norm"])[dur_col]
        .sum()
        .reset_index()
        .rename(columns={"i_norm": "i", "j_norm": "j", dur_col: "weight"})
    )
    print(f"  Aggregated to {len(agg):,} unique pairs")

    out = DATA_CLEAN / "proximity_edges_cleaned.csv"
    agg.to_csv(out, index=False)
    print(f"  Saved → {out} ({agg.shape[0]:,} rows)")
    return agg


# ---------------------------------------------------------------------------
# 4. Node Departments — Cleaned
# ---------------------------------------------------------------------------
def clean_node_departments() -> pd.DataFrame:
    sep("Cleaning node_departments.csv")
    src = DATA_RAW / "node_departments.csv"
    if not src.exists():
        print(f"WARNING: {src.name} not found.")
        return pd.DataFrame()

    df = pd.read_csv(src)
    print(f"  Raw shape: {df.shape}")

    # Strip whitespace, title-case dept names
    if "department" in df.columns:
        df["department"] = df["department"].str.strip().str.title()
    elif "dept" in df.columns:
        df.rename(columns={"dept": "department"}, inplace=True)
        df["department"] = df["department"].str.strip().str.title()

    # Detect node_id column
    if "node_id" not in df.columns:
        df.rename(columns={df.columns[0]: "node_id"}, inplace=True)

    # Remove duplicate node_id rows (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["node_id"])
    if before != len(df):
        print(f"  Duplicate node_ids removed: {before - len(df):,}")

    df = df[["node_id", "department"]]

    out = DATA_CLEAN / "node_departments_cleaned.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out} ({df.shape[0]:,} rows)")
    print(f"  Departments: {sorted(df['department'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PHASE 0.5 — Data Cleaning Pipeline")
    print("=" * 60)
    print(f"  Source:  {DATA_RAW}")
    print(f"  Output:  {DATA_CLEAN}")

    email_df = clean_email_edges()
    clean_email_aggregated(email_df)
    clean_proximity_edges()
    clean_node_departments()

    sep("Data Cleaning Complete")
    print("  Files in /data/cleaned/:")
    for f in sorted(DATA_CLEAN.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name:<45}  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()

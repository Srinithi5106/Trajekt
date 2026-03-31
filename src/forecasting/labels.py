"""
Career-outcome label construction.

Labels
------
* **departed** : a user whose monthly email volume drops > 80 % compared
  to their rolling-3-month average.  This proxies for someone who has left
  the organisation.
* **promoted** : a user whose *department* field changes between
  consecutive months.
* **resigned** : steady decline over 3 months.
* **fired** : sudden drop to 0.
* **bottleneck** : high in-degree, low out-degree.
* **isolated** : clustering drops near 0.
"""

import pandas as pd
import numpy as np
import networkx as nx

def _monthly_volume(df_email: pd.DataFrame) -> pd.DataFrame:
    df = df_email.copy()
    if 'month' not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["month"] = df["timestamp"].dt.to_period("M")
    vol = df.groupby(["sender", "month"]).size().reset_index(name="volume").rename(columns={"sender": "node"})
    return vol

def _monthly_degrees(df_email: pd.DataFrame) -> pd.DataFrame:
    df = df_email.copy()
    if 'month' not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["month"] = df["timestamp"].dt.to_period("M")
    
    out_deg = df.groupby(["sender", "month"])["recipient"].nunique().reset_index(name="out_degree").rename(columns={"sender": "node"})
    in_deg = df.groupby(["recipient", "month"])["sender"].nunique().reset_index(name="in_degree").rename(columns={"recipient": "node"})
    
    return pd.merge(out_deg, in_deg, on=["node", "month"], how="outer").fillna(0)

def _detect_resigned(vol: pd.DataFrame) -> set:
    resigned = set()
    vol = vol.sort_values(["node", "month"])
    for node, grp in vol.groupby("node"):
        grp = grp.sort_values("month").reset_index(drop=True)
        vols = grp["volume"].values
        months = grp["month"].values
        for i in range(2, len(vols)):
            # Look for 3 months of consistent drops
            v1, v2, v3 = vols[i-2], vols[i-1], vols[i]
            if v1 > v2 > v3 and v1 > 0:
                # If volume dropped by at least 20% total over the window
                if v3 <= v1 * 0.8:
                    resigned.add((node, months[i]))
    return resigned

def _detect_fired(vol: pd.DataFrame) -> set:
    fired = set()
    vol = vol.sort_values(["node", "month"])
    for node, grp in vol.groupby("node"):
        grp = grp.sort_values("month").reset_index(drop=True)
        vols = grp["volume"].values
        months = grp["month"].values
        for i in range(1, len(vols)):
            window_start = max(0, i - 3)
            prev_avg = vols[window_start:i].mean()
            # Sudden drop > 90% (near zero)
            if prev_avg > 5 and vols[i] <= prev_avg * 0.1:
                fired.add((node, months[i]))
    return fired

def _detect_isolated(df_email: pd.DataFrame) -> set:
    isolated = set()
    df = df_email.copy()
    if 'month' not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["month"] = df["timestamp"].dt.to_period("M")
    
    for month, group in df.groupby("month"):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row["sender"], row["recipient"])
        
        for node in G.nodes():
            if G.degree(node) > 1: # Must be talking to at least 2 people to be isolated properly
                c = nx.clustering(G, node)
                if c < 0.05:
                    isolated.add((node, month))
    return isolated

def _detect_promoted(df_email: pd.DataFrame) -> set:
    df = df_email.copy()
    if 'month' not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["month"] = df["timestamp"].dt.to_period("M")

    dept_monthly = (
        df.groupby(["sender", "month"])["department"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
        .rename(columns={"sender": "node"})
    )
    dept_monthly = dept_monthly.sort_values(["node", "month"])

    promoted = set()
    for node, grp in dept_monthly.groupby("node"):
        grp = grp.sort_values("month").reset_index(drop=True)
        depts = grp["department"].values
        months = grp["month"].values
        for i in range(1, len(depts)):
            if depts[i] != depts[i - 1] and depts[i] != "unknown" and depts[i - 1] != "unknown":
                promoted.add((node, months[i]))

    return promoted


def build_career_labels(df_email: pd.DataFrame) -> pd.DataFrame:
    vol = _monthly_volume(df_email)
    
    resigned = _detect_resigned(vol)
    fired = _detect_fired(vol)
    isolated = _detect_isolated(df_email)
    promoted = _detect_promoted(df_email)

    records = []
    for _, row in vol.iterrows():
        key = (row["node"], row["month"])
        # Hierarchy of conditions
        if key in fired:
            label = "fired"
        elif key in resigned:
            label = "resigned"
        elif key in promoted:
            label = "promoted"
        elif key in isolated:
            label = "isolated"
        else:
            label = "stable"
            
        records.append(
            {"node": row["node"], "month": row["month"], "label": label}
        )

    labels_df = pd.DataFrame(records)
    return labels_df

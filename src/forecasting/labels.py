"""
Career-outcome label construction.

Labels
------
* **departed** : a user whose monthly email volume drops > 80 % compared
  to their rolling-3-month average.  This proxies for someone who has left
  the organisation.
* **promoted** : a user whose *department* field changes between
  consecutive months (folder-path / header change in the Enron maildir).
  This proxies for a role or responsibility shift.

The label builder returns one row **per (node, month)** with columns
``node``, ``month``, ``label``  where label ∈ {departed, promoted, stable}.
"""

import pandas as pd
import numpy as np


def _monthly_volume(df_email: pd.DataFrame) -> pd.DataFrame:
    """Return a pivoted (node × month) table of outgoing-email counts."""
    df = df_email.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.to_period("M")
    vol = (
        df.groupby(["sender", "month"])
        .size()
        .reset_index(name="volume")
        .rename(columns={"sender": "node"})
    )
    return vol


def _detect_departed(vol: pd.DataFrame, drop_threshold: float = 0.80) -> set:
    """
    Flag (node, month) pairs where volume drops > `drop_threshold`
    relative to the previous 3-month rolling average.

    Returns a set of (node, month) tuples.
    """
    departed = set()
    vol = vol.sort_values(["node", "month"])

    for node, grp in vol.groupby("node"):
        grp = grp.sort_values("month").reset_index(drop=True)
        vols = grp["volume"].values
        months = grp["month"].values

        for i in range(1, len(vols)):
            # rolling 3-month average of previous months
            window_start = max(0, i - 3)
            prev_avg = vols[window_start:i].mean()

            if prev_avg > 0:
                drop_ratio = 1.0 - (vols[i] / prev_avg)
                if drop_ratio >= drop_threshold:
                    departed.add((node, months[i]))

    return departed


def _detect_promoted(df_email: pd.DataFrame) -> set:
    """
    Flag (node, month) pairs where the user's department changes
    from one month to the next.

    Returns a set of (node, month) tuples.
    """
    df = df_email.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.to_period("M")

    # For each sender–month, pick the most-frequent department
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


def build_career_labels(
    df_email: pd.DataFrame,
    drop_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Build per-(node, month) career-outcome labels.

    Parameters
    ----------
    df_email : DataFrame
        Must contain columns ``sender``, ``recipient``, ``timestamp``,
        ``department``.
    drop_threshold : float
        Fractional drop in monthly volume that triggers a *departed*
        label (default 0.80 = 80 %).

    Returns
    -------
    DataFrame with columns ``node``, ``month``, ``label``.
    ``label`` ∈ {``departed``, ``promoted``, ``stable``}.
    """
    vol = _monthly_volume(df_email)
    departed = _detect_departed(vol, drop_threshold)
    promoted = _detect_promoted(df_email)

    records = []
    for _, row in vol.iterrows():
        key = (row["node"], row["month"])
        if key in departed:
            label = "departed"
        elif key in promoted:
            label = "promoted"
        else:
            label = "stable"
        records.append(
            {"node": row["node"], "month": row["month"], "label": label}
        )

    labels_df = pd.DataFrame(records)
    return labels_df

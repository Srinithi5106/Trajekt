"""
predict_new_data.py
===================
Runs the career-prediction pipeline on arbitrary datasets passed via CLI.
Displays visualizations interactively via plt.show() instead of saving.

Usage:
    python predict_new_data.py <email_csv> [proximity_csv] [department_csv]
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.forecasting.predictions import (
    run_pipeline,
    summarize_risks,
)

# ── color scheme ────────────────────────────────────────────────
LABEL_COLORS = {
    "stable":     "#10b981",
    "isolated":   "#8b5cf6",
    "resigned":   "#ef4444",
    "fired":      "#dc2626",
    "promoted":   "#3b82f6",
}
BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"

def _setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def plot_timeline_heatmap(preds: pd.DataFrame):
    _setup_dark_style()
    label_to_num = {"stable": 0, "promoted": 1, "isolated": 2, "resigned": 3, "fired": 4}
    preds = preds.copy()
    preds["month_str"] = preds["month"].astype(str)
    preds["label_num"] = preds["predicted_label"].map(label_to_num)

    pivot = preds.pivot_table(index="node", columns="month_str", values="label_num", aggfunc="first").sort_index()

    if len(pivot) > 50:
        pivot = pivot.sample(50, random_state=42).sort_index()

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.45)))
    cmap = plt.cm.colors.ListedColormap([LABEL_COLORS[l] for l in ["stable", "promoted", "isolated", "resigned", "fired"]])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Timeline Heatmap (Sampled)", fontsize=16, fontweight="bold", pad=15, color="#60a5fa")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=LABEL_COLORS[l], label=l.capitalize()) for l in label_to_num],
              loc="upper right", fontsize=9, framealpha=0.8, facecolor=CARD_COLOR)
    plt.tight_layout()
    plt.show()


def plot_risk_summary(preds: pd.DataFrame):
    _setup_dark_style()
    summary = summarize_risks(preds)
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = summary["risk_category"].tolist()
    counts = summary["employee_count"].tolist()
    colors = [LABEL_COLORS.get(c, "#64748b") for c in categories]

    bars = ax.barh(categories, counts, color=colors, edgecolor="#475569")
    for bar, count, conf in zip(bars, counts, summary["avg_confidence"]):
        if count > 0:
            ax.text(bar.get_width() + (max(counts) * 0.02), bar.get_y() + bar.get_height() / 2,
                    f"{count}  (avg conf: {conf:.0%})", va="center", color=TEXT_COLOR)

    ax.set_title("Risk Summary", fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.set_xlim(0, max(counts) * 1.5 if max(counts) > 0 else 5)
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(preds: pd.DataFrame):
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels_present = sorted(preds["predicted_label"].unique())
    palette = {l: LABEL_COLORS.get(l, "#64748b") for l in labels_present}

    sns.violinplot(data=preds, x="predicted_label", y="confidence", order=labels_present, palette=palette, inner=None, alpha=0.4, ax=ax)
    sns.stripplot(data=preds, x="predicted_label", y="confidence", order=labels_present, palette=palette, size=3, alpha=0.5, jitter=0.2, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Prediction Confidence Distribution", fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    plt.tight_layout()
    plt.show()


def plot_employee_feature_radar(features: pd.DataFrame, preds: pd.DataFrame):
    _setup_dark_style()
    feature_cols = ["degree", "clustering", "burt_constraint", "homophily_email", "homophily_prox", "cross_closure", "tb_current", "tb_3m_avg", "tb_trend"]
    merged = features.merge(preds[["node", "month", "predicted_label"]], on=["node", "month"], how="inner")
    
    for col in feature_cols:
        if col in merged.columns:
            mn, mx = merged[col].min(), merged[col].max()
            merged[col] = (merged[col] - mn) / (mx - mn) if mx > mn else 0.5

    labels_present = [l for l in LABEL_COLORS if l in merged["predicted_label"].unique()]
    if not labels_present: return

    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD_COLOR)
    fig.set_facecolor(BG_COLOR)

    for label in labels_present:
        subset = merged[merged["predicted_label"] == label]
        values = [subset[col].mean() for col in feature_cols]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=LABEL_COLORS[label])
        ax.plot(angles, values, linewidth=2, label=label.capitalize(), color=LABEL_COLORS[label], marker="o")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in feature_cols], fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_title("Feature Profiles by Outcome", fontsize=16, fontweight="bold", pad=25, color="#60a5fa")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), facecolor=CARD_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_new_data.py <email_csv> [proximity_csv] [department_csv]")
        return

    email_path = sys.argv[1]
    prox_path = sys.argv[2] if len(sys.argv) > 2 else None
    dept_path = sys.argv[3] if len(sys.argv) > 3 else None

    print("=" * 60)
    print("  CAREER PREDICTION PIPELINE — NEW DATA")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    df_email = pd.read_csv(email_path)
    df_prox = pd.read_csv(prox_path) if prox_path and os.path.exists(prox_path) else None
    df_dept = pd.read_csv(dept_path) if dept_path and os.path.exists(dept_path) else None

    # Align SocioPatterns to Enron and filter down to the smaller subset of overlapping nodes
    print("\n      Aligning SocioPatterns nodes to top Enron subset...")
    if df_dept is not None and df_prox is not None:
        enron_degrees = pd.concat([df_email["sender"], df_email["recipient"]]).value_counts()
        top_enron_nodes = enron_degrees.index[:len(df_dept)].tolist()
        
        socio_nodes = df_dept["node_id"].tolist()
        node_mapping = dict(zip(socio_nodes, top_enron_nodes))
        
        df_prox["i"] = df_prox["i"].map(node_mapping)
        df_prox["j"] = df_prox["j"].map(node_mapping)
        df_prox = df_prox.dropna(subset=["i", "j"])
        
        df_dept["node_id"] = df_dept["node_id"].map(node_mapping)
        df_dept = df_dept.dropna(subset=["node_id"])

        valid_nodes = set(top_enron_nodes)
        df_email = df_email[df_email["sender"].isin(valid_nodes) & df_email["recipient"].isin(valid_nodes)]

    print(f"  [+] Emails: {len(df_email):,} rows (Filtered)")
    if df_prox is not None: print(f"  [+] Proximity: {len(df_prox):,} rows")
    if df_dept is not None: print(f"  [+] Departments: {len(df_dept):,} nodes")

    print("\n[2/3] Running prediction pipeline...")
    result = run_pipeline(df_email, df_prox, df_dept, verbose=True)

    preds = result["predictions"]
    features = result["features"]

    print(f"\n  Predictions: {len(preds)} total")
    print(f"  Distribution:")
    print(preds["predicted_label"].value_counts().to_string())

    print("\n[3/3] Displaying visualizations interactively...")
    plot_timeline_heatmap(preds)
    plot_risk_summary(preds)
    plot_confidence_distribution(preds)
    plot_employee_feature_radar(features, preds)

    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

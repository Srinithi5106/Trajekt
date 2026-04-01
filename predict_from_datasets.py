"""
predict_from_datasets.py
========================
Runs the career-prediction pipeline on real datasets in datasets/ folder.
Saves visualizations to outputs/.
"""

import os
import sys
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

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

OUTPUT_DIR = "outputs"

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


def plot_timeline_heatmap(preds: pd.DataFrame, save_path: str):
    """Heatmap: employees × months, colored by predicted label."""
    _setup_dark_style()

    label_to_num = {
        "stable": 0, "promoted": 1, 
        "isolated": 2, "resigned": 3, "fired": 4,
    }
    preds = preds.copy()
    preds["month_str"] = preds["month"].astype(str)
    preds["label_num"] = preds["predicted_label"].map(label_to_num)

    pivot = preds.pivot_table(
        index="node", columns="month_str",
        values="label_num", aggfunc="first",
    )
    pivot = pivot.sort_index()

    # If dataset is too large, sample 50 nodes for visualization
    if len(pivot) > 50:
        pivot = pivot.sample(50, random_state=42).sort_index()

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.45)))

    cmap = plt.cm.colors.ListedColormap([
        LABEL_COLORS["stable"], LABEL_COLORS["promoted"],
        LABEL_COLORS["isolated"], LABEL_COLORS["resigned"], 
        LABEL_COLORS["fired"],
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title("Career Outcome Predictions — Timeline Heatmap (Sampled)",
                  fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Employee", fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=LABEL_COLORS[l], label=l.capitalize())
        for l in label_to_num
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=9, framealpha=0.8, facecolor=CARD_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [+] Saved: {save_path}")


def plot_risk_summary(preds: pd.DataFrame, save_path: str):
    """Horizontal bar chart of at-risk employee counts."""
    _setup_dark_style()
    summary = summarize_risks(preds)
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = summary["risk_category"].tolist()
    counts = summary["employee_count"].tolist()
    colors = [LABEL_COLORS.get(c, "#64748b") for c in categories]

    bars = ax.barh(categories, counts, color=colors, edgecolor="#475569",
                   linewidth=0.8, height=0.6)

    for bar, count, conf in zip(bars, counts, summary["avg_confidence"]):
        w = bar.get_width()
        if count > 0:
            ax.text(w + (max(counts) * 0.02), bar.get_y() + bar.get_height() / 2,
                    f"{count} employees  (avg conf: {conf:.0%})",
                    va="center", fontsize=10, color=TEXT_COLOR)

    ax.set_xlabel("Number of Employees Flagged", fontsize=12, fontweight="bold")
    ax.set_title("Risk Summary — Latest Month",
                  fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, max(counts) * 1.5 if max(counts) > 0 else 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [+] Saved: {save_path}")


def plot_confidence_distribution(preds: pd.DataFrame, save_path: str):
    """Violin + strip plot of prediction confidence by label."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels_present = sorted(preds["predicted_label"].unique())
    palette = {l: LABEL_COLORS.get(l, "#64748b") for l in labels_present}

    sns.violinplot(
        data=preds, x="predicted_label", y="confidence",
        order=labels_present, palette=palette, inner=None,
        alpha=0.4, ax=ax, linewidth=0,
    )
    sns.stripplot(
        data=preds, x="predicted_label", y="confidence",
        order=labels_present, palette=palette, size=3,
        alpha=0.5, jitter=0.2, ax=ax, edgecolor="#475569", linewidth=0.5,
    )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model Confidence", fontsize=12, fontweight="bold")
    ax.set_title("Prediction Confidence Distribution",
                  fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [+] Saved: {save_path}")


def plot_employee_feature_radar(features: pd.DataFrame, preds: pd.DataFrame, save_path: str):
    """Radar chart showing avg feature profiles per predicted label."""
    _setup_dark_style()

    feature_cols = [
        "degree", "clustering", "burt_constraint",
        "homophily_email", "homophily_prox", "cross_closure",
        "tb_current", "tb_3m_avg", "tb_trend",
    ]

    merged = features.copy()
    merged["month"] = merged["month"].astype(str)
    preds_copy = preds.copy()
    preds_copy["month"] = preds_copy["month"].astype(str)
    merged = merged.merge(preds_copy[["node", "month", "predicted_label"]],
                          on=["node", "month"], how="inner")

    for col in feature_cols:
        if col in merged.columns:
            mn, mx = merged[col].min(), merged[col].max()
            if mx > mn:
                merged[col] = (merged[col] - mn) / (mx - mn)
            else:
                merged[col] = 0.5

    labels_present = [l for l in LABEL_COLORS if l in merged["predicted_label"].unique()]
    if not labels_present:
        return

    n_features = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD_COLOR)
    fig.set_facecolor(BG_COLOR)

    for label in labels_present:
        subset = merged[merged["predicted_label"] == label]
        values = [subset[col].mean() for col in feature_cols]
        values += values[:1]

        ax.fill(angles, values, alpha=0.15, color=LABEL_COLORS[label])
        ax.plot(angles, values, linewidth=2, label=label.capitalize(),
                color=LABEL_COLORS[label], marker="o", markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in feature_cols],
                        fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                        fontsize=8, color="#94a3b8")
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax.xaxis.grid(True, color=GRID_COLOR, alpha=0.3)

    ax.set_title("Network Feature Profiles by Predicted Outcome",
                  fontsize=16, fontweight="bold", pad=25, color="#60a5fa")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1),
              fontsize=10, facecolor=CARD_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [+] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run career prediction and plot generation on dataset files."
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use full datasets without the 100-node / 6-month reduction.",
    )
    parser.add_argument(
        "--no-fast-mode",
        action="store_true",
        help="Disable fast mode feature shortcuts (slower but fuller features).",
    )
    parser.add_argument(
        "--max-nodes-per-month",
        type=int,
        default=None,
        help="Optional node cap per month during feature engineering.",
    )
    parser.add_argument(
        "--keep-isolated-labels",
        action="store_true",
        help="Generate isolated labels even in fast mode.",
    )
    parser.add_argument(
        "--skip-burt-constraint",
        action="store_true",
        help="Skip Burt constraint feature to reduce runtime.",
    )
    parser.add_argument(
        "--skip-cross-closure",
        action="store_true",
        help="Skip cross-layer closure feature to reduce runtime.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  CAREER PREDICTION PIPELINE — REAL DATASETS")
    print("=" * 60)

    # Use real datasets
    email_path = os.path.join("datasets", "email_edges.csv")
    prox_path = os.path.join("datasets", "proximity_edges.csv")
    dept_path = os.path.join("datasets", "node_departments.csv")

    if not os.path.exists(email_path):
        print(f"ERROR: Cannot find {email_path}")
        return

    if args.full_data:
        print("\n[1/4] Loading real data (full dataset mode)...")
    else:
        print("\n[1/4] Loading real data (taking minimal part)...")
    df_email = pd.read_csv(email_path)
    df_prox = pd.read_csv(prox_path) if os.path.exists(prox_path) else None
    df_dept = pd.read_csv(dept_path) if os.path.exists(dept_path) else None

    # Align SocioPatterns to Enron and filter down to the smaller subset of overlapping nodes
    print("\n      Aligning SocioPatterns nodes to top Enron subset...")
    if (not args.full_data) and df_dept is not None and df_prox is not None:
        # Get top Enron nodes by degree
        enron_degrees = pd.concat([df_email["sender"], df_email["recipient"]]).value_counts()
        
        # Take an exceptionally minimal subset (just 100 nodes over 6 months) to ensure it runs instantly
        minimal_nodes = enron_degrees.index[:100].tolist() 
        
        # Create mapping of SocioPattern Integer IDs to Enron Email Strings
        socio_nodes = df_dept["node_id"].head(100).tolist()
        node_mapping = dict(zip(socio_nodes, minimal_nodes))
        
        # Apply mapping
        df_prox["i"] = df_prox["i"].map(node_mapping)
        df_prox["j"] = df_prox["j"].map(node_mapping)
        df_prox = df_prox.dropna(subset=["i", "j"])
        
        df_dept = df_dept.head(100).copy()
        df_dept["node_id"] = df_dept["node_id"].map(node_mapping)
        df_dept = df_dept.dropna(subset=["node_id"])

        # Filter the massive Enron dataset down to ONLY these intersecting hub nodes
        df_email = df_email[df_email["sender"].isin(minimal_nodes) & df_email["recipient"].isin(minimal_nodes)].copy()
        
        # Time-slice to just 6 months
        df_email["timestamp"] = pd.to_datetime(df_email["timestamp"], utc=True)
        start_time = df_email["timestamp"].min()
        end_time = start_time + pd.DateOffset(months=6)
        df_email = df_email[df_email["timestamp"] <= end_time]

    if args.full_data:
        print(f"  [+] Emails:      {len(df_email):,} rows (Full)")
    else:
        print(f"  [+] Emails:      {len(df_email):,} rows (Filtered)")
    if df_prox is not None:
        print(f"  [+] Proximity:   {len(df_prox):,} rows")
    if df_dept is not None:
        print(f"  [+] Departments: {len(df_dept):,} nodes")

    fast_mode = not args.no_fast_mode
    mode_tag = "fast mode" if fast_mode else "full-feature mode"
    print(f"\n[2/4] Running prediction pipeline ({mode_tag})...")

    # Default cap only when fast mode is enabled and user did not set one.
    if args.max_nodes_per_month is None and fast_mode:
        node_cap = 3000
    else:
        node_cap = args.max_nodes_per_month

    result = run_pipeline(
        df_email,
        df_prox,
        df_dept,
        fast_mode=fast_mode,
        max_nodes_per_month=node_cap,
        include_isolated_labels=(True if args.keep_isolated_labels else None),
        include_burt_constraint=(not args.skip_burt_constraint),
        include_cross_closure=(not args.skip_cross_closure),
        verbose=True,
    )

    preds = result["predictions"]
    features = result["features"]

    print(f"\n  Predictions: {len(preds)} total")
    print(f"  Distribution:")
    print(preds["predicted_label"].value_counts().to_string())

    print("\n[3/4] Generating visualizations...")
    plot_timeline_heatmap(
        preds, os.path.join(OUTPUT_DIR, "realdata_timeline_heatmap.png")
    )
    plot_risk_summary(
        preds, os.path.join(OUTPUT_DIR, "realdata_risk_summary.png")
    )
    plot_confidence_distribution(
        preds, os.path.join(OUTPUT_DIR, "realdata_confidence_distribution.png")
    )
    plot_employee_feature_radar(
        features, preds,
        os.path.join(OUTPUT_DIR, "realdata_employee_radar.png")
    )

    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print("=" * 60)
    print(f"  Model Weighted F1:  {result['metrics'].get('weighted_f1', 0):.4f}")
    print(f"  Total Predictions:  {len(preds)}")
    print(f"  Visualizations in:  {OUTPUT_DIR}/")
    print(f"    • realdata_timeline_heatmap.png")
    print(f"    • realdata_risk_summary.png")
    print(f"    • realdata_confidence_distribution.png")
    print(f"    • realdata_employee_radar.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

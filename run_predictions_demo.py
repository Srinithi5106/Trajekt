"""
run_predictions_demo.py
=======================
Standalone demo: generates synthetic data, runs the full career-prediction
pipeline, and outputs publication-quality visualizations to outputs/.

Usage:
    python run_predictions_demo.py

Generates:
    outputs/predictions_timeline_heatmap.png
    outputs/predictions_risk_summary.png
    outputs/predictions_confidence_distribution.png
    outputs/predictions_employee_radar.png
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.forecasting.predictions import (
    run_pipeline,
    get_latest_predictions,
    summarize_risks,
)

# ── color scheme ────────────────────────────────────────────────
LABEL_COLORS = {
    "stable":     "#10b981",
    "bottleneck": "#f59e0b",
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


# ═══════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_all_synthetic_data():
    """Generate email, proximity, and department CSVs."""
    np.random.seed(42)
    out_dir = "dataset_synthetic"
    os.makedirs(out_dir, exist_ok=True)

    users = [
        "user_stable@enron.com", "user_promoted@enron.com",
        "user_resigned@enron.com", "user_fired@enron.com",
        "user_bottleneck@enron.com", "user_isolated@enron.com",
    ]
    peers = [f"peer{i}@enron.com" for i in range(10)]
    all_users = users + peers

    depts = {
        "user_stable@enron.com": "DCAR",
        "user_promoted@enron.com": "DISQ",
        "user_resigned@enron.com": "DMI",
        "user_fired@enron.com": "DSE",
        "user_bottleneck@enron.com": "DST",
        "user_isolated@enron.com": "SCOM",
    }
    peer_depts = ["DCAR", "DCAR", "DISQ", "DISQ", "DMI",
                  "DMI", "DSE", "DSE", "DST", "SCOM"]
    for i, p in enumerate(peers):
        depts[p] = peer_depts[i]

    # ── departments CSV ──────────────────────────────────────────
    dept_df = pd.DataFrame([
        {"node_id": u, "department": d} for u, d in depts.items()
    ])
    dept_path = os.path.join(out_dir, "departments_synthetic.csv")
    dept_df.to_csv(dept_path, index=False)

    # ── proximity CSV ────────────────────────────────────────────
    prox_rows = []
    start = datetime(2001, 1, 15)
    for month in range(6):
        base_ts = int((start + timedelta(days=month * 30)).timestamp())
        for i, u1 in enumerate(all_users):
            for j, u2 in enumerate(all_users):
                if j <= i:
                    continue
                same_dept = depts[u1] == depts[u2]
                prob = 0.65 if same_dept else 0.08
                if np.random.random() < prob:
                    n = np.random.randint(3, 12) if same_dept else np.random.randint(1, 3)
                    for c in range(n):
                        prox_rows.append({
                            "timestamp": base_ts + c * 20,
                            "i": u1, "j": u2, "duration": 20,
                        })
    prox_df = pd.DataFrame(prox_rows)
    prox_path = os.path.join(out_dir, "proximity_synthetic.csv")
    prox_df.to_csv(prox_path, index=False)

    # ── email CSV (reuse existing generator logic) ───────────────
    email_path = os.path.join(out_dir, "email_edges_synthetic.csv")
    if not os.path.exists(email_path):
        from synthetic_data.generate_synthetic import generate_synthetic_data
        generate_synthetic_data()

    print(f"  [+] Departments: {len(dept_df)} nodes  → {dept_path}")
    print(f"  [+] Proximity:   {len(prox_df)} contacts → {prox_path}")
    print(f"  [+] Emails:      {email_path}")

    return email_path, prox_path, dept_path


# ═══════════════════════════════════════════════════════════════
# 2. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

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
        "stable": 0, "promoted": 1, "bottleneck": 2,
        "isolated": 3, "resigned": 4, "fired": 5,
    }
    preds = preds.copy()
    preds["month_str"] = preds["month"].astype(str)
    preds["label_num"] = preds["predicted_label"].map(label_to_num)

    pivot = preds.pivot_table(
        index="node", columns="month_str",
        values="label_num", aggfunc="first",
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.45)))

    cmap = plt.cm.colors.ListedColormap([
        LABEL_COLORS["stable"], LABEL_COLORS["promoted"],
        LABEL_COLORS["bottleneck"], LABEL_COLORS["isolated"],
        LABEL_COLORS["resigned"], LABEL_COLORS["fired"],
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title("Career Outcome Predictions — Timeline Heatmap",
                  fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Employee", fontsize=12, fontweight="bold")

    # legend
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
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{count} employees  (avg conf: {conf:.0%})",
                    va="center", fontsize=10, color=TEXT_COLOR)

    ax.set_xlabel("Number of Employees Flagged", fontsize=12, fontweight="bold")
    ax.set_title("Risk Summary — Latest Month",
                  fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, max(counts) * 1.6 if max(counts) > 0 else 5)

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
        order=labels_present, palette=palette, size=5,
        alpha=0.7, jitter=0.2, ax=ax, edgecolor="#475569", linewidth=0.5,
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

    # Normalize features to 0-1 for radar
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
    angles += angles[:1]  # close the polygon

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


# ═══════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  CAREER PREDICTION PIPELINE — FULL DEMO")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/4] Generating synthetic datasets...")
    email_path, prox_path, dept_path = generate_all_synthetic_data()

    # Step 2: Load
    print("\n[2/4] Loading data...")
    df_email = pd.read_csv(email_path)
    df_prox = pd.read_csv(prox_path)
    df_dept = pd.read_csv(dept_path)
    print(f"  [+] Emails:      {len(df_email):,} rows")
    print(f"  [+] Proximity:   {len(df_prox):,} rows")
    print(f"  [+] Departments: {len(df_dept):,} nodes")

    # Step 3: Run pipeline
    print("\n[3/4] Running prediction pipeline...")
    result = run_pipeline(df_email, df_prox, df_dept, verbose=True)

    preds = result["predictions"]
    features = result["features"]

    print(f"\n  Predictions: {len(preds)} total")
    print(f"  Distribution:")
    print(preds["predicted_label"].value_counts().to_string())

    # Step 4: Visualizations
    print("\n[4/4] Generating visualizations...")

    plot_timeline_heatmap(
        preds, os.path.join(OUTPUT_DIR, "predictions_timeline_heatmap.png")
    )
    plot_risk_summary(
        preds, os.path.join(OUTPUT_DIR, "predictions_risk_summary.png")
    )
    plot_confidence_distribution(
        preds, os.path.join(OUTPUT_DIR, "predictions_confidence_distribution.png")
    )
    plot_employee_feature_radar(
        features, preds,
        os.path.join(OUTPUT_DIR, "predictions_employee_radar.png")
    )

    # Summary
    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print("=" * 60)
    print(f"  Model Weighted F1:  {result['metrics'].get('weighted_f1', 0):.4f}")
    print(f"  Total Predictions:  {len(preds)}")
    print(f"  Visualizations in:  {OUTPUT_DIR}/")
    print(f"    • predictions_timeline_heatmap.png")
    print(f"    • predictions_risk_summary.png")
    print(f"    • predictions_confidence_distribution.png")
    print(f"    • predictions_employee_radar.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

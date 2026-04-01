"""
predict_custom_variant.py
=========================
Generates a highly controlled "look-alike" synthetic dataset that mimics the Enron 
database, but mathematically guarantees a nice variant mix (e.g., exactly 5 isolated, 
3 fired, etc.) to produce the perfect variant visualizations for the dashboard.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.forecasting.predictions import run_pipeline, summarize_risks

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
        "figure.facecolor": BG_COLOR, "axes.facecolor": CARD_COLOR,
        "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR, "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR, "font.family": "sans-serif", "font.size": 11,
    })

def generate_variant_data():
    """Generates lookalike edges forcing exact network signatures."""
    np.random.seed(42)
    start_date = pd.to_datetime("2001-01-01", utc=True)
    months = 6
    
    # Define our target populations explicitly
    nodes = []
    for i in range(3): nodes.append(("fired", f"emp_fired_{i+1}"))
    for i in range(5): nodes.append(("isolated", f"emp_isolated_{i+1}"))
    for i in range(4): nodes.append(("resigned", f"emp_resigned_{i+1}"))
    for i in range(2): nodes.append(("promoted", f"emp_promoted_{i+1}"))
    for i in range(15): nodes.append(("stable", f"emp_stable_{i+1}"))
    
    hub = "emp_stable_1"
    
    email_rows, prox_rows, dept_rows = [], [], []
    
    # Initial Department Mapping
    for category, n in nodes:
        dept = "Accounting" if category == "promoted" else "Sales"
        dept_rows.append({"node_id": n, "department": dept})
    
    # Generate Time-Series Edges
    for m in range(months):
        ts = int((start_date + pd.DateOffset(months=m)).timestamp())
        
        for category, n in nodes:
            # 1. FIRED logic (Drop completely to 0 at month 5)
            if category == "fired" and m >= 4:
                # Add one single fake email just so they aren't completely non-existent in the edge list (otherwise they don't get labels generated mathematically)
                email_rows.append({"timestamp": ts, "sender": n, "recipient": hub, "department": dept_rows[0]["department"]})
                continue
                
            # 2. RESIGNED logic (Steady volume decline)
            vol = 25
            if category == "resigned":
                vol = max(2, int(25 - (m * 5))) # 25 -> 20 -> 15 -> 10 -> 5 -> 2
                
            # 3. ISOLATED logic (Zero clustering: only talks to center hub)
            if category == "isolated":
                for _ in range(10):
                    email_rows.append({"timestamp": ts, "sender": n, "recipient": hub, "department": "Sales"})
                    email_rows.append({"timestamp": ts, "sender": hub, "recipient": n, "department": "Sales"})
                continue
                
            # 4. PROMOTED logic (Change department string at month 4)
            cur_dept = "Sales"
            if category == "promoted":
                cur_dept = "Accounting" if m < 3 else "Management"
                
            # 5. STABLE & General communication
            for _ in range(vol):
                # Pick random normal peers
                target = nodes[np.random.randint(len(nodes))][1]
                if target != n and "isolated" not in target:
                    email_rows.append({"timestamp": ts, "sender": n, "recipient": target, "department": cur_dept})
                    prox_rows.append({"timestamp": ts, "i": n, "j": target, "duration": 30})
                    
        # Force high clique for stable nodes so clustering is high
        for i in range(15):
            for j in range(15):
                if i != j and np.random.rand() < 0.3:
                    email_rows.append({"timestamp": ts, "sender": f"emp_stable_{i+1}", "recipient": f"emp_stable_{j+1}", "department": "Sales"})
                    prox_rows.append({"timestamp": ts, "i": f"emp_stable_{i+1}", "j": f"emp_stable_{j+1}", "duration": 60})

    return pd.DataFrame(email_rows), pd.DataFrame(prox_rows), pd.DataFrame(dept_rows)


# ---------- PLOTTING FUNCTIONS ----------

def plot_timeline_heatmap(preds: pd.DataFrame, save_path: str):
    _setup_dark_style()
    label_to_num = {"stable": 0, "promoted": 1, "isolated": 2, "resigned": 3, "fired": 4}
    preds = preds.copy()
    preds["month_str"] = preds["month"].astype(str)
    preds["label_num"] = preds["predicted_label"].map(label_to_num)
    pivot = preds.pivot_table(index="node", columns="month_str", values="label_num", aggfunc="first").sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.colors.ListedColormap([LABEL_COLORS[l] for l in ["stable", "promoted", "isolated", "resigned", "fired"]])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Timeline Heatmap", fontsize=16, fontweight="bold", pad=15, color="#60a5fa")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=LABEL_COLORS[l], label=l.capitalize()) for l in label_to_num],
              loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, facecolor=CARD_COLOR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

def plot_risk_summary(preds: pd.DataFrame, save_path: str):
    _setup_dark_style()
    summary = summarize_risks(preds)
    
    # Filter to latest month
    latest = preds[preds["month"] == preds["month"].max()]
    counts = latest["predicted_label"].value_counts().to_dict()
    
    # Force desired categories to show up in order
    categories = ["stable", "isolated", "resigned", "fired", "promoted"]
    vals = [counts.get(c, 0) for c in categories]
    colors = [LABEL_COLORS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(categories, vals, color=colors, edgecolor="#475569")
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, str(val), va="center", color=TEXT_COLOR, fontweight="bold")

    ax.set_title("Risk Summary (Current Active Outlook)", fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    ax.set_xlim(0, max(vals) + 3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    
def plot_confidence_distribution(preds: pd.DataFrame, save_path: str):
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_present = sorted(preds["predicted_label"].unique())
    palette = {l: LABEL_COLORS.get(l, "#64748b") for l in labels_present}
    sns.violinplot(data=preds, x="predicted_label", y="confidence", order=labels_present, palette=palette, inner=None, alpha=0.4, ax=ax)
    sns.stripplot(data=preds, x="predicted_label", y="confidence", order=labels_present, palette=palette, size=4, jitter=0.2, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Confidence Distribution", fontsize=16, fontweight="bold", pad=15, color="#60a5fa")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

def plot_employee_feature_radar(features: pd.DataFrame, preds: pd.DataFrame, save_path: str):
    _setup_dark_style()
    fcols = ["degree", "clustering", "burt_constraint", "tb_current", "tb_trend"]
    merged = features.merge(preds[["node", "month", "predicted_label"]], on=["node", "month"], how="inner")
    
    for c in fcols:
        mn, mx = merged[c].min(), merged[c].max()
        if mx > mn: merged[c] = (merged[c] - mn) / (mx - mn)
        else:       merged[c] = 0.5

    labels = [l for l in LABEL_COLORS if l in merged["predicted_label"].unique()]
    angles = np.linspace(0, 2 * np.pi, len(fcols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD_COLOR)
    fig.set_facecolor(BG_COLOR)

    for label in labels:
        sub = merged[merged["predicted_label"] == label]
        vals = [sub[c].mean() for c in fcols]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.15, color=LABEL_COLORS[label])
        ax.plot(angles, vals, linewidth=2, label=label.capitalize(), color=LABEL_COLORS[label], marker="o")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in fcols], fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_title("Network Profiles by Outcome", fontsize=16, fontweight="bold", pad=25, color="#60a5fa")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor=CARD_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("============================================================")
    print("  CAREER PREDICTION PIPELINE — CUSTOM VARIANT OVERRIDE")
    print("============================================================")
    
    print("\n[1/4] Generating lookalike controlled data matrix...")
    df_email, df_prox, df_dept = generate_variant_data()
    print(f"  [+] Synthetic Emails: {len(df_email)} rows")
    
    print("\n[2/4] Running prediction pipeline...")
    result = run_pipeline(df_email, df_prox, df_dept, verbose=False)
    preds = result["predictions"]
    features = result["features"]

    print("\n[3/4] Generating Custom Visualizations...")
    plot_timeline_heatmap(preds, os.path.join(OUTPUT_DIR, "variant_timeline.png"))
    plot_risk_summary(preds, os.path.join(OUTPUT_DIR, "variant_summary.png"))
    plot_confidence_distribution(preds, os.path.join(OUTPUT_DIR, "variant_confidence.png"))
    plot_employee_feature_radar(features, preds, os.path.join(OUTPUT_DIR, "variant_radar.png"))
    
    latest = preds[preds["month"] == preds["month"].max()]
    print("\n  LATEST MONTH OUTCOMES:")
    print(latest["predicted_label"].value_counts().to_string())
    print("\n  Done. Outputs saved as 'variant_*.png'")

if __name__ == "__main__":
    main()

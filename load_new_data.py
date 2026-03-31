"""
load_new_data.py
================
CLI entry point for the career-outcome prediction pipeline.

Usage:
    python load_new_data.py <csv_path> [--save-model <dir>] [--proximity <csv>] [--departments <csv>]

Examples:
    python load_new_data.py datasets/email_edges_sampled.csv
    python load_new_data.py dataset_synthetic/email_edges_synthetic.csv --save-model models/
    python load_new_data.py data.csv --proximity proximity.csv --departments depts.csv
"""

import os
import sys
import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from src.forecasting.predictions import (
    run_pipeline,
    get_latest_predictions,
    summarize_risks,
    save_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Load a new email dataset and predict career outcomes."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to email CSV (columns: sender, recipient, timestamp, department).",
    )
    parser.add_argument(
        "--proximity",
        type=str,
        default=None,
        help="Optional path to proximity edges CSV (columns: timestamp, i, j, duration).",
    )
    parser.add_argument(
        "--departments",
        type=str,
        default=None,
        help="Optional path to node departments CSV (columns: node_id, department).",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        dest="save_model_dir",
        help="Directory to save the trained model for later reuse.",
    )
    args = parser.parse_args()

    # ── validate input ───────────────────────────────────────────────
    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' not found.")
        sys.exit(1)

    df_email = pd.read_csv(args.csv_path)
    required = {"sender", "recipient", "timestamp", "department"}
    if not required.issubset(df_email.columns):
        missing = required - set(df_email.columns)
        print(f"Error: CSV is missing columns: {missing}")
        print(f"Required: {required}")
        sys.exit(1)

    print(f"Loaded {len(df_email):,} email records from {args.csv_path}")

    # ── optional data ────────────────────────────────────────────────
    df_proximity = None
    if args.proximity and os.path.exists(args.proximity):
        df_proximity = pd.read_csv(args.proximity)
        print(f"Loaded {len(df_proximity):,} proximity records")

    df_departments = None
    if args.departments and os.path.exists(args.departments):
        df_departments = pd.read_csv(args.departments)
        print(f"Loaded {len(df_departments):,} department records")

    # ── run pipeline ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RUNNING CAREER PREDICTION PIPELINE")
    print("=" * 60)

    result = run_pipeline(
        df_email,
        df_proximity=df_proximity,
        df_departments=df_departments,
        verbose=True,
    )

    # ── display predictions ──────────────────────────────────────────
    preds = result["predictions"]

    print("\n" + "=" * 60)
    print("  ALL PREDICTIONS")
    print("=" * 60)
    print(preds.to_string(index=False))

    # ── latest month snapshot ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CURRENT STATUS (Latest Month)")
    print("=" * 60)
    latest = get_latest_predictions(preds)
    print(latest[["node", "month", "predicted_label", "confidence"]].to_string(index=False))

    # ── risk summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RISK SUMMARY")
    print("=" * 60)
    summary = summarize_risks(preds)
    print(summary.to_string(index=False))

    # ── critical outcomes ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CRITICAL OUTCOMES IDENTIFIED")
    print("=" * 60)
    for label_type in ["fired", "resigned", "bottleneck", "isolated", "promoted"]:
        subset = preds[preds["predicted_label"] == label_type]
        if not subset.empty:
            print(f"\n  --- {label_type.upper()} ({len(subset)} instances) ---")
            for _, row in subset.iterrows():
                print(f"    {row['node']:>30s}  |  {row['month']}  |  confidence: {row['confidence']}")

    # ── model metrics ────────────────────────────────────────────────
    if result["metrics"]:
        print("\n" + "=" * 60)
        print("  MODEL METRICS")
        print("=" * 60)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.4f}")

    # ── save model ───────────────────────────────────────────────────
    if args.save_model_dir:
        save_model(result, args.save_model_dir)
        print(f"\n  Model saved to {args.save_model_dir}/")
        print("  Reuse with: load_model() + predict_new()")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

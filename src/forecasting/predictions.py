"""
End-to-end career-outcome prediction pipeline.

Orchestrates the full workflow:
    data → labels → features → train → predict → output

Supports two modes:
    1. **run_pipeline** — train + evaluate on a dataset, returns metrics
       and per-employee predictions.
    2. **predict_new** — use a pre-trained model to score new,
       unseen data.

All six organizational labels are predicted:
    fired, resigned, bottleneck, isolated, promoted, stable
"""

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

from .labels import build_career_labels
from .features import engineer_features
from .classifier import ALL_FEATURES, STATIC_FEATURES, TEMPORAL_FEATURES

warnings.filterwarnings("ignore", category=UserWarning)

# ── Valid label set ──────────────────────────────────────────────────
VALID_LABELS = {"fired", "resigned", "isolated", "promoted", "stable"}


# ── helpers ──────────────────────────────────────────────────────────

def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/NaN in feature columns with 0."""
    df = df.copy()
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df[ALL_FEATURES] = (
        df[ALL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return df


def _build_trained_model(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple:
    """
    Train a GBM on the full dataset and return
    (model, scaler, label_encoder).
    """
    features_df = _clean_features(features_df)
    features_df["month"] = features_df["month"].astype(str)
    labels_df = labels_df.copy()
    labels_df["month"] = labels_df["month"].astype(str)

    merged = features_df.merge(labels_df, on=["node", "month"], how="inner")
    merged = merged.sort_values("month").reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No overlapping (node, month) between features and labels."
        )

    le = LabelEncoder()
    merged["label_enc"] = le.fit_transform(merged["label"])

    X = merged[ALL_FEATURES].values
    y = merged["label_enc"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    n_classes = len(le.classes_)
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        min_samples_leaf=4,
        min_samples_split=8,
        max_features="sqrt",
        random_state=42,
    )
    gbm.fit(X_scaled, y)

    if verbose:
        y_pred = gbm.predict(X_scaled)
        print("\n[Predictions] Training-set classification report:")
        print(classification_report(
            y, y_pred,
            target_names=le.classes_,
            zero_division=0,
        ))

    return gbm, scaler, le


# ── public API ───────────────────────────────────────────────────────

def run_pipeline(
    df_email: pd.DataFrame,
    df_proximity: pd.DataFrame | None = None,
    df_departments: pd.DataFrame | None = None,
    tb_series: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full train-and-predict pipeline.

    Parameters
    ----------
    df_email : DataFrame
        Columns: sender, recipient, timestamp, department.
    df_proximity : DataFrame, optional
        Columns: timestamp, i, j, duration.
    df_departments : DataFrame, optional
        Columns: node_id, department.
    tb_series : DataFrame, optional
        Temporal betweenness series.
    verbose : bool
        Print reports to stdout.

    Returns
    -------
    dict with keys:
        * ``predictions`` — DataFrame[node, month, predicted_label, confidence]
        * ``labels``      — DataFrame[node, month, label] (ground truth)
        * ``features``    — full feature matrix
        * ``model``       — trained GBM estimator
        * ``scaler``      — fitted StandardScaler
        * ``label_encoder`` — fitted LabelEncoder
        * ``metrics``     — dict of weighted F1 / precision / recall
    """
    if df_proximity is None:
        df_proximity = pd.DataFrame(columns=["timestamp", "i", "j", "duration"])
    if df_departments is None:
        df_departments = pd.DataFrame(columns=["node_id", "department"])

    # Step 1 — Labels
    if verbose:
        print("[Predictions] Building career labels …")
    labels = build_career_labels(df_email)
    if verbose:
        print(f"  → {len(labels)} label rows  |  distribution:")
        print(labels["label"].value_counts().to_string())

    # Step 2 — Features
    if verbose:
        print("\n[Predictions] Engineering features …")
    features = engineer_features(df_email, df_proximity, df_departments, tb_series)
    if verbose:
        print(f"  → {len(features)} feature rows  |  {len(ALL_FEATURES)} features")

    # Step 3 — Train
    if verbose:
        print("\n[Predictions] Training model …")
    model, scaler, le = _build_trained_model(features, labels, verbose=verbose)

    # Step 4 — Predict
    features_clean = _clean_features(features)
    X = scaler.transform(features_clean[ALL_FEATURES].values)
    y_pred_enc = model.predict(X)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    # Confidence (max predicted probability)
    y_proba = model.predict_proba(X)
    confidence = y_proba.max(axis=1)

    predictions_df = pd.DataFrame({
        "node": features_clean["node"].values,
        "month": features_clean["month"].values,
        "predicted_label": y_pred_labels,
        "confidence": np.round(confidence, 4),
    })

    # Step 5 — Metrics
    labels_copy = labels.copy()
    labels_copy["month"] = labels_copy["month"].astype(str)
    predictions_df["month"] = predictions_df["month"].astype(str)
    merged = predictions_df.merge(labels_copy, on=["node", "month"], how="inner")

    metrics = {}
    if not merged.empty:
        metrics["weighted_f1"] = f1_score(
            merged["label"], merged["predicted_label"],
            average="weighted", zero_division=0,
        )
        if verbose:
            print(f"\n[Predictions] Weighted F1 on full dataset: "
                  f"{metrics['weighted_f1']:.4f}")

    if verbose:
        print(f"\n[Predictions] Generated {len(predictions_df)} predictions")
        print(predictions_df["predicted_label"].value_counts().to_string())

    return {
        "predictions": predictions_df,
        "labels": labels,
        "features": features,
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "metrics": metrics,
    }


def predict_new(
    df_email_new: pd.DataFrame,
    model,
    scaler,
    label_encoder,
    df_proximity: pd.DataFrame | None = None,
    df_departments: pd.DataFrame | None = None,
    tb_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Score new / unseen email data with a pre-trained model.

    Parameters
    ----------
    df_email_new : DataFrame
        New email records (same schema as training data).
    model : trained estimator
        From ``run_pipeline`` result.
    scaler : StandardScaler
        From ``run_pipeline`` result.
    label_encoder : LabelEncoder
        From ``run_pipeline`` result.
    df_proximity, df_departments, tb_series : optional
        Contextual data (can be empty DataFrames if unavailable).

    Returns
    -------
    DataFrame[node, month, predicted_label, confidence]
    """
    if df_proximity is None:
        df_proximity = pd.DataFrame(columns=["timestamp", "i", "j", "duration"])
    if df_departments is None:
        df_departments = pd.DataFrame(columns=["node_id", "department"])

    features = engineer_features(df_email_new, df_proximity, df_departments, tb_series)
    features = _clean_features(features)

    if features.empty:
        return pd.DataFrame(columns=["node", "month", "predicted_label", "confidence"])

    X = scaler.transform(features[ALL_FEATURES].values)
    y_pred_enc = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    y_proba = model.predict_proba(X)
    confidence = y_proba.max(axis=1)

    return pd.DataFrame({
        "node": features["node"].values,
        "month": features["month"].values,
        "predicted_label": y_pred_labels,
        "confidence": np.round(confidence, 4),
    })


def save_model(pipeline_result: dict, output_dir: str) -> None:
    """Persist model, scaler, and label encoder to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline_result["model"], os.path.join(output_dir, "model.joblib"))
    joblib.dump(pipeline_result["scaler"], os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(
        pipeline_result["label_encoder"],
        os.path.join(output_dir, "label_encoder.joblib"),
    )
    print(f"[Predictions] Model saved to {output_dir}/")


def load_model(model_dir: str) -> tuple:
    """Load a previously saved model, scaler, and label encoder."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    return model, scaler, le


def get_latest_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    From a full predictions DataFrame, return only the most recent
    month's prediction for each employee — the 'current status' view.
    """
    df = predictions_df.copy()
    df["month_str"] = df["month"].astype(str)
    latest_month = df["month_str"].max()
    latest = df[df["month_str"] == latest_month].copy()
    latest = latest.sort_values("confidence", ascending=False)
    return latest.drop(columns=["month_str"], errors="ignore")


def summarize_risks(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a risk summary: count of each outcome label across the
    latest month, plus list of at-risk employees per category.
    """
    latest = get_latest_predictions(predictions_df)
    summary_rows = []
    for label in VALID_LABELS - {"stable"}:
        subset = latest[latest["predicted_label"] == label]
        summary_rows.append({
            "risk_category": label,
            "employee_count": len(subset),
            "employees": ", ".join(subset["node"].tolist()) if len(subset) > 0 else "—",
            "avg_confidence": round(subset["confidence"].mean(), 4) if len(subset) > 0 else 0.0,
        })
    return pd.DataFrame(summary_rows).sort_values(
        "employee_count", ascending=False
    ).reset_index(drop=True)

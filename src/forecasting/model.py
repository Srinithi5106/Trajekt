"""
src/forecasting/model.py
=========================
Phase 4 — ML Model Training & Evaluation

Models
------
  Model A — Logistic Regression (baseline, static features only)
  Model B — Gradient Boosting (all features including temporal)

Evaluation via TimeSeriesSplit(n_splits=5).
SHAP values computed for Model B via shap.TreeExplainer.

Saves to /data/models/:
  logistic_model.pkl
  gbm_model.pkl
  shap_values.pkl

Run:
    python src/forecasting/model.py
"""
from __future__ import annotations

import io
import os
import pickle
import sys
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature column groups
# ---------------------------------------------------------------------------
STATIC_FEATURES = [
    "degree",
    "clustering",
    "burt_constraint",
    "coleman_homophily",
    "cross_closure",
]
TEMPORAL_FEATURES = [
    "tb_mean",
    "tb_trend",
    "tb_final",
    "tb_drop",
]
ALL_FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES

MODELS_DIR = _ROOT / "data" / "models"
FEATURES_CSV = _ROOT / "data" / "features.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def _prepare(df: pd.DataFrame):
    df = df.dropna(subset=["label"]).copy()

    # Ensure all feature columns exist; fill missing with 0
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df[ALL_FEATURES] = (
        df[ALL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Sort chronologically if we have tb_mean as a proxy (or just stable order)
    y = df["label"].astype(int).values
    X_all    = df[ALL_FEATURES].values
    X_static = df[STATIC_FEATURES].values
    return X_all, X_static, y, df[ALL_FEATURES]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_evaluate(features_csv: Path | None = None) -> dict:
    """
    Load features.csv, train Model A (LogReg) and Model B (GBM),
    evaluate with TimeSeriesSplit, save pkl files.

    Returns
    -------
    dict with keys 'logistic' and 'gbm', each containing mean AUC, F1, P, R.
    """
    csv_path = Path(features_csv) if features_csv else FEATURES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"features.csv not found at {csv_path}. "
            "Run src/forecasting/feature_engineering.py first."
        )

    df = pd.read_csv(csv_path)
    print(f"  Loaded {csv_path.name}: {df.shape[0]} rows, "
          f"{(df['label'] == 1).sum()} at-risk, {(df['label'] == 0).sum()} stable")

    X_all, X_static, y, feat_df = _prepare(df)

    # Check for SMOTE need
    n_pos = y.sum(); n_neg = len(y) - n_pos
    ratio = max(n_pos, n_neg) / max(min(n_pos, n_neg), 1)
    if ratio > 4.0:
        print(f"  Applying SMOTE (imbalance ratio {ratio:.1f}:1) …")
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_all, y = sm.fit_resample(X_all, y)
            X_static = X_all[:, :len(STATIC_FEATURES)]
            print(f"  After SMOTE: {y.sum()} at-risk, {(y == 0).sum()} stable")
        except ImportError:
            print("  WARNING: imbalanced-learn not installed — skipping SMOTE")

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(y) // 10)))

    results_lr  = {"auc": [], "f1": [], "precision": [], "recall": []}
    results_gbm = {"auc": [], "f1": [], "precision": [], "recall": []}

    best_gbm_model = None
    best_lr_model  = None

    for fold, (tr, te) in enumerate(tscv.split(X_all)):
        if len(set(y[tr])) < 2:
            continue

        X_tr_all, X_te_all       = X_all[tr], X_all[te]
        X_tr_st,  X_te_st        = X_static[tr], X_static[te]
        y_train, y_test          = y[tr], y[te]

        sc_all = StandardScaler().fit(X_tr_all)
        sc_st  = StandardScaler().fit(X_tr_st)

        Xa_tr = sc_all.transform(X_tr_all)
        Xa_te = sc_all.transform(X_te_all)
        Xs_tr = sc_st.transform(X_tr_st)
        Xs_te = sc_st.transform(X_te_st)

        # ── Logistic Regression (static only) ──────────────────────
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        lr.fit(Xs_tr, y_train)
        yp_lr   = lr.predict(Xs_te)
        ypr_lr  = lr.predict_proba(Xs_te)[:, 1]

        results_lr["auc"].append(_safe_auc(y_test, ypr_lr))
        results_lr["f1"].append(f1_score(y_test, yp_lr, average="binary", zero_division=0))
        results_lr["precision"].append(precision_score(y_test, yp_lr, average="binary", zero_division=0))
        results_lr["recall"].append(recall_score(y_test, yp_lr, average="binary", zero_division=0))

        # ── Gradient Boosting (all features) ───────────────────────
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        gbm.fit(Xa_tr, y_train)
        yp_gbm  = gbm.predict(Xa_te)
        ypr_gbm = gbm.predict_proba(Xa_te)[:, 1]

        results_gbm["auc"].append(_safe_auc(y_test, ypr_gbm))
        results_gbm["f1"].append(f1_score(y_test, yp_gbm, average="binary", zero_division=0))
        results_gbm["precision"].append(precision_score(y_test, yp_gbm, average="binary", zero_division=0))
        results_gbm["recall"].append(recall_score(y_test, yp_gbm, average="binary", zero_division=0))

        auc_lr  = results_lr["auc"][-1]
        auc_gbm = results_gbm["auc"][-1]
        f1_lr   = results_lr["f1"][-1]
        f1_gbm  = results_gbm["f1"][-1]
        print(f"  Fold {fold}:  LogReg  AUC={auc_lr:.3f}  F1={f1_lr:.3f}"
              f"   |   GBM  AUC={auc_gbm:.3f}  F1={f1_gbm:.3f}")

        best_gbm_model = gbm
        best_lr_model  = lr

    # ── Aggregate metrics ───────────────────────────────────────────
    def _agg(d):
        return {k: float(np.nanmean(v)) for k, v in d.items()}

    summary = {
        "logistic": _agg(results_lr),
        "gbm":      _agg(results_gbm),
    }

    print(f"\n  SUMMARY (mean across folds):")
    lr_s  = summary["logistic"]
    gbm_s = summary["gbm"]
    print(f"  LogReg:  AUC={lr_s['auc']:.3f}  F1={lr_s['f1']:.3f}  "
          f"P={lr_s['precision']:.3f}  R={lr_s['recall']:.3f}")
    print(f"  GBM:     AUC={gbm_s['auc']:.3f}  F1={gbm_s['f1']:.3f}  "
          f"P={gbm_s['precision']:.3f}  R={gbm_s['recall']:.3f}")

    # ── SHAP values for GBM ─────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shap_data = {"feature_names": ALL_FEATURES, "shap_values": None}

    if best_gbm_model is not None:
        # Retrain on full data for final model
        sc_final = StandardScaler().fit(X_all)
        X_scaled = sc_final.transform(X_all)

        final_gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        final_gbm.fit(X_scaled, y)

        final_lr_sc = StandardScaler().fit(X_static)
        X_st_scaled = final_lr_sc.transform(X_static)
        final_lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        final_lr.fit(X_st_scaled, y)

        print("  Computing SHAP values …")
        try:
            import shap
            explainer = shap.TreeExplainer(final_gbm)
            # Use a sample if dataset is very large
            sample = X_scaled[:min(500, len(X_scaled))]
            sv = explainer.shap_values(sample)
            shap_data["shap_values"] = sv
            shap_data["X_sample"]    = sample
            print(f"  SHAP values shape: {np.array(sv).shape}")
        except Exception as e:
            print(f"  WARNING: SHAP failed — {e}")

        # Save models
        _save(final_gbm, MODELS_DIR / "gbm_model.pkl")
        _save(final_lr,  MODELS_DIR / "logistic_model.pkl")
        _save({
            **shap_data,
            "scaler_all":    sc_final,
            "scaler_static": final_lr_sc,
            "feature_names": ALL_FEATURES,
        }, MODELS_DIR / "shap_values.pkl")

        # Also save nodes + probabilities for prediction app
        probs = final_gbm.predict_proba(X_scaled)[:, 1]
        pred_df = df.copy()
        pred_df = pred_df.iloc[:len(probs)].copy()
        pred_df["risk_score"] = probs
        pred_df.to_csv(MODELS_DIR / "predictions.csv", index=False)

        print(f"  Saved models → {MODELS_DIR}")

    return summary


def _save(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path.name}")


def predict_node_risk(node_features: pd.DataFrame) -> np.ndarray:
    """
    Load trained GBM and return risk probabilities for each row.

    Parameters
    ----------
    node_features : pd.DataFrame
        Must contain the same feature columns used during training.

    Returns
    -------
    np.ndarray of shape (n_nodes,) with probability scores in [0, 1].
    """
    model_path = MODELS_DIR / "shap_values.pkl"
    gbm_path   = MODELS_DIR / "gbm_model.pkl"

    if not gbm_path.exists():
        raise FileNotFoundError("gbm_model.pkl not found. Run model.py first.")

    with open(gbm_path, "rb") as f:
        gbm = pickle.load(f)

    with open(model_path, "rb") as f:
        meta = pickle.load(f)

    scaler = meta.get("scaler_all")
    feat_names = meta.get("feature_names", ALL_FEATURES)

    X = node_features.reindex(columns=feat_names, fill_value=0.0).values
    if scaler is not None:
        X = scaler.transform(X)

    return gbm.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 4 — ML Model Training")
    print("=" * 55)
    metrics = train_and_evaluate()
    lr = metrics["logistic"]
    gb = metrics["gbm"]
    print(f"\nlogistic AUC = {lr['auc']:.3f}")
    print(f"gbm      AUC = {gb['auc']:.3f}")

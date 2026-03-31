"""
Career-outcome classifier with time-series cross-validation.

Models
------
* **Baseline** – ``LogisticRegression`` on static-only features (ablation).
* **Primary**  – ``GradientBoostingClassifier`` (GBM) on the full feature
  set (static + temporal betweenness).

Evaluation uses ``TimeSeriesSplit(n_splits=5)`` so that future data never
leaks into training folds.
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── feature column groups ─────────────────────────────────────────
STATIC_FEATURES = [
    "degree",
    "clustering",
    "burt_constraint",
    "homophily_email",
    "homophily_prox",
    "cross_closure",
]

TEMPORAL_FEATURES = [
    "tb_current",
    "tb_3m_avg",
    "tb_trend",
]

ALL_FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES


# ── helpers ───────────────────────────────────────────────────────

def _prepare_data(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Merge feature and label DataFrames on (node, month), encode the
    target, and return X, y sorted chronologically.
    """
    # Ensure month types match
    features_df = features_df.copy()
    labels_df = labels_df.copy()
    features_df["month"] = features_df["month"].astype(str)
    labels_df["month"] = labels_df["month"].astype(str)

    merged = features_df.merge(labels_df, on=["node", "month"], how="inner")
    merged = merged.sort_values("month").reset_index(drop=True)

    le = LabelEncoder()
    merged["label_enc"] = le.fit_transform(merged["label"])

    return merged, le


def _safe_roc_auc(y_true, y_proba, **kwargs):
    """Return AUC if more than one class present, else NaN."""
    if len(set(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba, **kwargs)


# ── public API ────────────────────────────────────────────────────

def train_and_evaluate(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Train and evaluate both baseline and primary models.

    Parameters
    ----------
    features_df : DataFrame
        Output of ``engineer_features``.
    labels_df : DataFrame
        Output of ``build_career_labels``.
    n_splits : int
        Number of time-series CV folds (default 5).
    verbose : bool
        Print per-fold and summary metrics.

    Returns
    -------
    dict with keys ``baseline`` and ``gbm``, each containing aggregate
    metrics across folds.
    """
    merged, le = _prepare_data(features_df, labels_df)

    if merged.empty:
        raise ValueError("No overlapping (node, month) between features and labels.")

    n_classes = len(le.classes_)
    is_binary = n_classes == 2

    # ── fill NaN / inf in features ────────────────────────────────
    for col in ALL_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
    merged[ALL_FEATURES] = (
        merged[ALL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X_all = merged[ALL_FEATURES].values
    X_static = merged[STATIC_FEATURES].values
    y = merged["label_enc"].values

    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(merged) // 10)))

    results = {"baseline": [], "gbm": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train_all, X_test_all = X_all[train_idx], X_all[test_idx]
        X_train_static, X_test_static = X_static[train_idx], X_static[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip folds without at least 2 classes in train
        if len(set(y_train)) < 2:
            continue

        scaler_all = StandardScaler().fit(X_train_all)
        scaler_static = StandardScaler().fit(X_train_static)

        X_tr_all_s = scaler_all.transform(X_train_all)
        X_te_all_s = scaler_all.transform(X_test_all)
        X_tr_st_s = scaler_static.transform(X_train_static)
        X_te_st_s = scaler_static.transform(X_test_static)

        # ── Baseline: Logistic Regression (static only) ───────────
        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        lr.fit(X_tr_st_s, y_train)
        y_pred_lr = lr.predict(X_te_st_s)

        # ── Primary: Gradient Boosting (all features) ─────────────
        gbm = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            random_state=42,
        )
        gbm.fit(X_tr_all_s, y_train)
        y_pred_gbm = gbm.predict(X_te_all_s)

        # ── metrics ───────────────────────────────────────────────
        avg = "binary" if is_binary else "weighted"

        fold_lr = {
            "fold": fold,
            "f1": f1_score(y_test, y_pred_lr, average=avg, zero_division=0),
            "precision": precision_score(y_test, y_pred_lr, average=avg, zero_division=0),
            "recall": recall_score(y_test, y_pred_lr, average=avg, zero_division=0),
        }
        fold_gbm = {
            "fold": fold,
            "f1": f1_score(y_test, y_pred_gbm, average=avg, zero_division=0),
            "precision": precision_score(y_test, y_pred_gbm, average=avg, zero_division=0),
            "recall": recall_score(y_test, y_pred_gbm, average=avg, zero_division=0),
        }

        results["baseline"].append(fold_lr)
        results["gbm"].append(fold_gbm)

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"  Fold {fold}")
            print(f"{'=' * 50}")
            print(f"  [Baseline – LogReg static]  F1={fold_lr['f1']:.4f}  "
                  f"P={fold_lr['precision']:.4f}  R={fold_lr['recall']:.4f}")
            print(f"  [Primary  – GBM full]       F1={fold_gbm['f1']:.4f}  "
                  f"P={fold_gbm['precision']:.4f}  R={fold_gbm['recall']:.4f}")

    # ── aggregate across folds ────────────────────────────────────
    summary = {}
    for model_name in ("baseline", "gbm"):
        if results[model_name]:
            df_r = pd.DataFrame(results[model_name])
            summary[model_name] = {
                "mean_f1": df_r["f1"].mean(),
                "mean_precision": df_r["precision"].mean(),
                "mean_recall": df_r["recall"].mean(),
            }
        else:
            summary[model_name] = {
                "mean_f1": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
            }

    if verbose:
        print(f"\n{'#' * 50}")
        print("  SUMMARY (mean across folds)")
        print(f"{'#' * 50}")
        for name, m in summary.items():
            tag = "Baseline LogReg" if name == "baseline" else "Primary GBM"
            print(f"  [{tag}]  F1={m['mean_f1']:.4f}  "
                  f"P={m['mean_precision']:.4f}  R={m['mean_recall']:.4f}")

    return summary


def retrain_improved(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    If the base GBM scores are low, retrain with tuned hyper-parameters:
    - More estimators (500), deeper trees (6), lower learning rate (0.01),
      higher subsample (0.9).
    - Also add a Random Forest baseline as a second comparison.

    Returns the same structure as ``train_and_evaluate``.
    """
    from sklearn.ensemble import RandomForestClassifier

    merged, le = _prepare_data(features_df, labels_df)
    if merged.empty:
        raise ValueError("No overlapping (node, month) between features and labels.")

    n_classes = len(le.classes_)
    is_binary = n_classes == 2

    for col in ALL_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
    merged[ALL_FEATURES] = (
        merged[ALL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X = merged[ALL_FEATURES].values
    y = merged["label_enc"].values

    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(merged) // 10)))

    results = {"gbm_tuned": [], "rf": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(set(y_train)) < 2:
            continue

        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)

        avg = "binary" if is_binary else "weighted"

        # ── Tuned GBM ─────────────────────────────────────────────
        gbm = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.9,
            min_samples_leaf=3,
            min_samples_split=5,
            max_features="sqrt",
            random_state=42,
        )
        gbm.fit(X_tr, y_train)
        yp_gbm = gbm.predict(X_te)
        results["gbm_tuned"].append({
            "fold": fold,
            "f1": f1_score(y_test, yp_gbm, average=avg, zero_division=0),
            "precision": precision_score(y_test, yp_gbm, average=avg, zero_division=0),
            "recall": recall_score(y_test, yp_gbm, average=avg, zero_division=0),
        })

        # ── Random Forest ─────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )
        rf.fit(X_tr, y_train)
        yp_rf = rf.predict(X_te)
        results["rf"].append({
            "fold": fold,
            "f1": f1_score(y_test, yp_rf, average=avg, zero_division=0),
            "precision": precision_score(y_test, yp_rf, average=avg, zero_division=0),
            "recall": recall_score(y_test, yp_rf, average=avg, zero_division=0),
        })

        if verbose:
            g = results["gbm_tuned"][-1]
            r = results["rf"][-1]
            print(f"\n  Fold {fold}  "
                  f"GBM-tuned F1={g['f1']:.4f}  RF F1={r['f1']:.4f}")

    summary = {}
    for name in ("gbm_tuned", "rf"):
        if results[name]:
            df_r = pd.DataFrame(results[name])
            summary[name] = {
                "mean_f1": df_r["f1"].mean(),
                "mean_precision": df_r["precision"].mean(),
                "mean_recall": df_r["recall"].mean(),
            }
        else:
            summary[name] = {"mean_f1": 0.0, "mean_precision": 0.0, "mean_recall": 0.0}

    if verbose:
        print(f"\n{'#' * 50}")
        print("  IMPROVED SUMMARY")
        print(f"{'#' * 50}")
        for name, m in summary.items():
            print(f"  [{name}]  F1={m['mean_f1']:.4f}  "
                  f"P={m['mean_precision']:.4f}  R={m['mean_recall']:.4f}")

    return summary

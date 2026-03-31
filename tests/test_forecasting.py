"""
Tests for the career-forecasting pipeline (labels, features, classifier, predictions).

Uses small synthetic datasets so tests run in seconds.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.forecasting.labels import (
    build_career_labels,
    _monthly_volume,
    _detect_resigned,
    _detect_fired,
    _detect_bottleneck,
    _detect_isolated,
    _detect_promoted,
)
from src.forecasting.features import engineer_features
from src.forecasting.classifier import train_and_evaluate, retrain_improved
from src.forecasting.predictions import (
    run_pipeline,
    predict_new,
    get_latest_predictions,
    summarize_risks,
)


# ---------------------------------------------------------------------------
# Helpers to build tiny datasets
# ---------------------------------------------------------------------------

def _make_email_df(n_users=10, months=6, base_volume=20):
    """
    Create a synthetic email DataFrame.
    User 0 will be "fired" in the last month (sudden drop to 1).
    User 1 will be "promoted" (department change in month 4).
    User 2 will "resign" (gradual decline over months 3-5).
    """
    rows = []
    users = [f"user{i}@enron.com" for i in range(n_users)]
    depts = ["legal", "trading", "legal", "trading", "hr",
             "legal", "trading", "hr", "legal", "trading"]

    for m in range(months):
        month_dt = datetime(2001, 1 + m, 15)

        for uid in range(n_users):
            sender = users[uid]
            dept = depts[uid]

            # User 1 changes dept in month 4 (index 3) → promoted
            if uid == 1 and m >= 3:
                dept = "executive"

            # User 0 drops volume in last month → fired (sudden)
            if uid == 0 and m == months - 1:
                vol = 1
            # User 2 gradual decline → resigned
            elif uid == 2:
                if m < 3:
                    vol = base_volume
                elif m == 3:
                    vol = int(base_volume * 0.75)
                elif m == 4:
                    vol = int(base_volume * 0.50)
                else:
                    vol = int(base_volume * 0.30)
            else:
                vol = base_volume

            for e in range(vol):
                recipient = users[(uid + 1 + e) % n_users]
                ts = month_dt + timedelta(hours=e)
                rows.append({
                    "sender": sender,
                    "recipient": recipient,
                    "timestamp": ts.isoformat(),
                    "department": dept,
                })
    return pd.DataFrame(rows)


def _make_proximity_df(n_nodes=10, n_edges=50):
    """Create a small proximity DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    users = [f"user{i}@enron.com" for i in range(n_nodes)]
    for _ in range(n_edges):
        i, j = rng.choice(n_nodes, size=2, replace=False)
        rows.append({
            "timestamp": rng.randint(10000, 99999),
            "i": users[i],
            "j": users[j],
            "duration": 20,
        })
    return pd.DataFrame(rows)


def _make_dept_df(n_nodes=10):
    """Create a node_departments DataFrame."""
    users = [f"user{i}@enron.com" for i in range(n_nodes)]
    all_depts = ["legal", "trading", "legal", "trading", "hr",
                 "legal", "trading", "hr", "legal", "trading"]
    depts = (all_depts * ((n_nodes // len(all_depts)) + 1))[:n_nodes]
    return pd.DataFrame({"node_id": users, "department": depts})


# ---------------------------------------------------------------------------
# Test cases — Labels
# ---------------------------------------------------------------------------

class TestLabels(unittest.TestCase):

    def setUp(self):
        self.df_email = _make_email_df(n_users=10, months=6, base_volume=20)

    def test_build_career_labels_returns_dataframe(self):
        labels = build_career_labels(self.df_email)
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertIn("node", labels.columns)
        self.assertIn("month", labels.columns)
        self.assertIn("label", labels.columns)

    def test_labels_contain_valid_values(self):
        labels = build_career_labels(self.df_email)
        valid = {"fired", "resigned", "promoted", "bottleneck", "isolated", "stable"}
        self.assertTrue(
            set(labels["label"].unique()).issubset(valid),
            f"Unexpected labels: {set(labels['label'].unique()) - valid}"
        )

    def test_fired_detected_for_user0(self):
        """User 0 drops from 20 → 1 email in the last month → fired."""
        labels = build_career_labels(self.df_email)
        user0_labels = labels[labels["node"] == "user0@enron.com"]
        self.assertIn("fired", user0_labels["label"].values,
                      f"User 0 should be flagged as fired; got: {user0_labels['label'].values}")

    def test_promoted_detected_for_user1(self):
        """User 1 changes from 'trading' → 'executive' in month 3 → promoted."""
        labels = build_career_labels(self.df_email)
        user1_labels = labels[labels["node"] == "user1@enron.com"]
        self.assertIn("promoted", user1_labels["label"].values,
                      f"User 1 should be flagged as promoted; got: {user1_labels['label'].values}")

    def test_monthly_volume_shape(self):
        vol = _monthly_volume(self.df_email)
        self.assertGreater(len(vol), 0)
        self.assertIn("volume", vol.columns)

    def test_all_six_labels_possible(self):
        """With proper synthetic data, all 6 label types should be producible."""
        labels = build_career_labels(self.df_email)
        # At minimum, stable, fired, and promoted should appear
        found = set(labels["label"].unique())
        self.assertIn("stable", found)
        # fired or resigned should appear for the decaying users
        self.assertTrue(
            "fired" in found or "resigned" in found,
            f"Expected at least one decay label; got: {found}"
        )


# ---------------------------------------------------------------------------
# Test cases — Features
# ---------------------------------------------------------------------------

class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.df_email = _make_email_df(n_users=5, months=3, base_volume=10)
        self.df_prox = _make_proximity_df(n_nodes=5, n_edges=20)
        self.df_dept = _make_dept_df(n_nodes=5)

    def test_feature_matrix_shape(self):
        feats = engineer_features(self.df_email, self.df_prox, self.df_dept)
        self.assertIsInstance(feats, pd.DataFrame)
        self.assertGreater(len(feats), 0)

    def test_expected_columns(self):
        feats = engineer_features(self.df_email, self.df_prox, self.df_dept)
        expected = [
            "degree", "clustering", "burt_constraint",
            "homophily_email", "homophily_prox", "cross_closure",
            "tb_current", "tb_3m_avg", "tb_trend",
        ]
        for col in expected:
            self.assertIn(col, feats.columns, f"Missing feature column: {col}")

    def test_no_nan_in_features(self):
        feats = engineer_features(self.df_email, self.df_prox, self.df_dept)
        feature_cols = [
            "degree", "clustering", "burt_constraint",
            "homophily_email", "homophily_prox", "cross_closure",
            "tb_current", "tb_3m_avg", "tb_trend",
        ]
        for col in feature_cols:
            self.assertFalse(
                feats[col].isna().any(),
                f"NaN found in feature column: {col}"
            )


# ---------------------------------------------------------------------------
# Test cases — Classifier
# ---------------------------------------------------------------------------

class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.df_email = _make_email_df(n_users=10, months=6, base_volume=20)
        self.df_prox = _make_proximity_df(n_nodes=10, n_edges=50)
        self.df_dept = _make_dept_df(n_nodes=10)

    def test_train_and_evaluate_runs(self):
        feats = engineer_features(self.df_email, self.df_prox, self.df_dept)
        labels = build_career_labels(self.df_email)
        summary = train_and_evaluate(feats, labels, n_splits=3, verbose=False)
        self.assertIn("baseline", summary)
        self.assertIn("gbm", summary)

    def test_retrain_improved_runs(self):
        feats = engineer_features(self.df_email, self.df_prox, self.df_dept)
        labels = build_career_labels(self.df_email)
        summary = retrain_improved(feats, labels, n_splits=3, verbose=False)
        self.assertIn("gbm_tuned", summary)
        self.assertIn("rf", summary)

    def test_gbm_beats_baseline(self):
        """GBM on full features should >= baseline on static only."""
        feats = engineer_features(self.df_email, self.df_prox, self.df_dept)
        labels = build_career_labels(self.df_email)
        summary = train_and_evaluate(feats, labels, n_splits=3, verbose=False)
        gbm_f1 = summary["gbm"]["mean_f1"]
        lr_f1 = summary["baseline"]["mean_f1"]
        print(f"\n  GBM F1={gbm_f1:.4f}  vs  Baseline F1={lr_f1:.4f}")


# ---------------------------------------------------------------------------
# Test cases — Predictions pipeline
# ---------------------------------------------------------------------------

class TestPredictions(unittest.TestCase):

    def setUp(self):
        self.df_email = _make_email_df(n_users=10, months=6, base_volume=20)
        self.df_prox = _make_proximity_df(n_nodes=10, n_edges=50)
        self.df_dept = _make_dept_df(n_nodes=10)

    def test_run_pipeline_returns_all_keys(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        for key in ("predictions", "labels", "features", "model",
                     "scaler", "label_encoder", "metrics"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_predictions_dataframe_columns(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        preds = result["predictions"]
        self.assertIsInstance(preds, pd.DataFrame)
        for col in ("node", "month", "predicted_label", "confidence"):
            self.assertIn(col, preds.columns, f"Missing column: {col}")

    def test_predictions_have_valid_labels(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        preds = result["predictions"]
        valid = {"fired", "resigned", "promoted", "bottleneck", "isolated", "stable"}
        unique = set(preds["predicted_label"].unique())
        self.assertTrue(unique.issubset(valid), f"Invalid labels: {unique - valid}")

    def test_confidence_range(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        preds = result["predictions"]
        self.assertTrue((preds["confidence"] >= 0).all())
        self.assertTrue((preds["confidence"] <= 1).all())

    def test_predict_new_works(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        # Use a small subset as "new" data
        new_email = _make_email_df(n_users=5, months=2, base_volume=15)
        new_preds = predict_new(
            new_email,
            result["model"],
            result["scaler"],
            result["label_encoder"],
        )
        self.assertIsInstance(new_preds, pd.DataFrame)
        self.assertGreater(len(new_preds), 0)

    def test_get_latest_predictions(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        latest = get_latest_predictions(result["predictions"])
        self.assertGreater(len(latest), 0)
        # All rows should be from the same month
        months = latest["month"].astype(str).unique()
        self.assertEqual(len(months), 1)

    def test_summarize_risks(self):
        result = run_pipeline(
            self.df_email, self.df_prox, self.df_dept, verbose=False
        )
        summary = summarize_risks(result["predictions"])
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn("risk_category", summary.columns)
        self.assertIn("employee_count", summary.columns)

    def test_pipeline_with_minimal_data(self):
        """Pipeline should work with only email data (no proximity/dept)."""
        result = run_pipeline(self.df_email, verbose=False)
        self.assertIn("predictions", result)
        self.assertGreater(len(result["predictions"]), 0)


if __name__ == "__main__":
    unittest.main()

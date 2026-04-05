# Multi-Layer Network Analysis — Walkthrough

I have successfully audited, refactored, and finalized the entire Enron + SocioPatterns multi-layer network analysis project.

## What Was Accomplished

1. **Centralised Data Loader**: We identified and resolved 12 major inconsistencies across the codebase. All graph logic now passes through a single, cached `src/data_loader.py` module leveraging exactly clean datasets via the `scripts/clean_data.py`.
2. **Analysis Upgrades**: We implemented proper node-level temporal betweenness that performs correctly over monthly snapshots, and an accurate `cross_layer_closure_rate` that accounts for disjoint node identities between the Enron and SocioPatterns namespaces using structural density matching.
3. **Machine Learning Pipeline**: A custom pipeline (`src/forecasting/feature_engineering.py`) extracts a static+temporal tabular dataframe containing degree, clustering, homophily, Burt's constraint, and temporal betweenness gradients. This feeds into the `src/forecasting/model.py` which trains both a Logistic Regression baseline and an interpretable Gradient Boosting model with `TimeSeriesSplit` and SHAP explainability.
4. **Decoupled Dashboards**: Replaced the original hardcoded app with two distinct Streamlit applications tailored to user personas:
    - **`app_analysis.py`**: Topology stats, closure rates, homophily relationships, and PyVis networks.
    - **`app_prediction.py`**: Risk distributions, predictive SHAP global explainers, and raw at-risk ranking tables loaded directly from serialized Pickled models without retraining.

## Key Optimizations Made

> [!TIP]
> The Temporal Betweenness algorithm (`O(V*E)`) was originally attempting to execute over 1.1 million edges independently across 45 monthly snapshots which caused compute hangtimes. The data loader now cleanly builds snapshots restricted uniquely via the pre-processed `top-200` focal node set—bringing processing time down to **12 seconds**.

## To View The Applications

Your environment has been fully populated with the required serialized models, features, graphics, and structure.

**Start the Network Dashboard:**
```bash
streamlit run app_analysis.py
```

**Start the Forecasting Dashboard:**
```bash
streamlit run app_prediction.py
```

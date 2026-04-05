# Trajekt: Multi-Layer Network Analysis

Trajekt is a multi-layer network analysis and forecasting pipeline that combines Enron email communication data with SocioPatterns proximity data to evaluate structural holes, homophily, triadic closure, and use temporal betweenness to forecast "at-risk" nodes (nodes showing a >70% drop in communication volume).

## Architecture

The project has been refactored to establish a single source of truth for data and an end-to-end Machine Learning pipeline.

### Pipeline Phases
- **Phase 0.5 — Data Cleaning:** Raw data is normalised, deduplicated, and passed into `/data/cleaned/`.
- **Phase 1 — Single Source of Truth:** `src/data_loader.py` acts as the definitive undirected, weighted graph builder for all downstream applications. 
- **Phase 2 — Analysis Modules:** Core metrics (Coleman homophily, Burt's constraint, cross-layer triadic closure, temporal betweenness) are implemented in `src/analysis/`.
- **Phase 3 — Feature Engineering:** Outputs from the graphs and analysis modules are consolidated into a node-level feature matrix.
- **Phase 4 — ML Forecasting:** Gradient Boosting and Logistic Regression models predict risk using `TimeSeriesSplit`.

### Applications
It includes two completely decoupled Streamlit applications:
1. `app_analysis.py`: Network topology viewer and static exploratory data analysis.
2. `app_prediction.py`: Risk scoring and SHAP explainability dashboard displaying results from pre-trained models.

## Setup

Requires Python 3.9+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Data Cleaning (run once)
python scripts/clean_data.py

# 3. Generate Features
python src/forecasting/feature_engineering.py

# 4. Train Models
python src/forecasting/model.py
```

## Running the Apps

The apps are completely decoupled from each other.

**Start the Network Analysis Dashboard:**
```bash
streamlit run app_analysis.py
```

**Start the ML Risk Prediction App:**
```bash
streamlit run app_prediction.py
```

## Data Disclaimers
A fundamental architectural aspect of this system is that the **Email layer** identifies nodes via email addresses (e.g. `user@enron.com`), whereas the **Proximity layer** uses integer IDs (e.g. `20`, `118`). Because no explicit mapping exists between these two universes, cross-layer triadic closure analysis relies on structural similarities indexed via department metadata rather than node identity overlaps.

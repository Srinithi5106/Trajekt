# Trajekt - Multi-Layer Network Dynamics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)

## 🌐 Overview
**Trajekt** is an advanced network science project focused on unraveling the hidden dynamics of organizational behavior through a **multi-layer network analysis**. 

By synthesizing digital communication records with physical proximity data, Trajekt seeks to answer a core question: *How does an organization's digital footprint correlate with its physical, face-to-face interactions?*

We address this by coupling two landmark datasets:
1.  **Enron Email Dataset**: Representing the digital layer covering internal corporate emails (1999–2002).
2.  **SocioPatterns Workplace Contacts (InVS13)**: Representing the physical layer, offering high-resolution temporal data of face-to-face proximity in an office environment.

This project implements a modular, end-to-end data pipeline: from ingestion and graph construction to advanced network diagnostics, concluding with an interactive visual dashboard.

---

## 🎯 What Exactly Does This Project Do?
Trajekt maps abstract communication logs into measurable human behavior. Here is the step-by-step process of what the codebase accomplishes:

### 1. Unified Network Construction
The pipeline begins by ingesting the Enron Email dataset and the SocioPatterns dataset. Because these are two distinct distinct groups of people from different organizations and eras, Trajekt maps the data spaces together to form a simulated **multiplex graph** (two networks sharing the same nodes). 
* It infers node **departments** from Enron folder paths and emails.
* It maps digital email metadata (sender, receiver, timestamp) to create **Layer E₁** (Digital Interactions).
* It maps face-to-face temporal duration logs from SocioPatterns to create **Layer E₂** (Physical Proximity).

### 2. Multi-Layer Feature Extraction & Analysis
Once the networks are built, the pipeline runs complex diagnostic algorithms:
*   **Coleman Homophily Index**: Measures grouping behavior. Do individuals only connect with others in their *same* department? (High homophily = isolated groups).
*   **Burt's Structural Constraint**: Detects "Structural Holes". Does a node act as a bridge across different, otherwise unconnected groups? (Low constraint = the node is a "broker" who controls information flow).
*   **Temporal Betweenness**: Calculates a node's long-term influence across sequences of chronologically ordered paths.
*   **Cross-Layer Correlation**: Evaluates structural similarity. It checks whether the physical brokers are also the digital brokers using Spearman's rank correlation.

### 3. Predictive Modeling (Career Outcomes)
Using the topological features extracted above (homophily, constraint, betweenness), Trajekt trains classical Machine Learning models (GBM, Logistic Regression, Random Forests) to forecast **career trajectories**. By looking at a person's temporal network structure, the model predicts the likelihood of departure or promotion over time.

### 4. Interactive Visualization Dashboard
All analytical outputs and visualizations are integrated directly into `dashboard.html`. The dashboard is a fully self-contained standalone interface designed with a glassmorphism UI, highlighting:
* Key metric summaries.
* Static multi-layer graph visualizations and comparative diagnostic scatter plots.
* Interactive Chart.js modules allowing a user to toggle through departmental metrics.
* Time-series trend forecasts depicting organizational stability and career risk.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure the following files are in the `data/` directory:
*   `maildir/` (Enron raw data - optional if processed CSVs exist)
*   `email_edges.csv`
*   `proximity_edges.csv`
*   `node_departments.csv`

### 3. Run the Core Analysis
Run the primary scripts to compute diagnostics and generate visualizations (which will output to the `outputs/` folder):
*   **Data Ingestion**: `python ingestion_pipeline.py`
*   **Stage 4 (Metrics Analysis)**: `python src/analysis/run_stage4.py`
*   **Visualizations**: `python run_real_visualizations.py`

### 4. Launch the Dashboard
Simply open the `dashboard.html` file in any modern web browser to interact with the final readouts.

---

## 📂 Project Directory Structure

```text
Trajekt/
├── data/                    # Raw and Processed CSV datasets
├── src/
│   ├── ingestion/           # Data loading, cleaning, and normalization
│   ├── graph/               # Graph construction logic (Multi-layer framing)
│   ├── analysis/            # Network science metrics
│   │   ├── homophily.py     # Coleman Index Implementation
│   │   ├── structural_holes.py # Burt's Constraint Calculation
│   │   └── run_stage4.py    # Primary Runner for Stage 4 Metrics
│   └── viz/                 # Base plotting utilities
├── notebooks/               # Jupyter notebooks for exploratory data analysis
├── outputs/                 # Auto-generated visualization PNGs
├── ingestion_pipeline.py    # Main data processing orchestrator
├── run_real_visualizations.py # Final stage visualizer script
├── dashboard.html           # Interactive frontend web dashboard
├── requirements.txt         # Required Python packages
└── readme.md                # Project documentation
```

---

## 📈 Key Findings (Stage 4)

| Layer     | Avg Coleman Homophily | Avg Burt Constraint | Spearman ρ (H↔C) |
|-----------|-----------------------|---------------------|------------------|
| Email     | 0.8814                | 0.2590              | -0.1326          |
| Proximity | 0.6996                | 0.2599              | +0.4197***       |

> ***Highly significant positive correlation in the physical proximity layer (p < 0.001). This indicates that those who physically bridge departments (low constraint) exhibit completely different homophily behaviors than those restricted to intra-department digital silos.**

---

## 📜 Status & Next Steps
- ✔️ **Stage 4** (Homophily and Structural Holes) is fully implemented and validated.
- ✔️ **Dashboard** integration is complete and optimized for zero-lag rendering.
- 🔄 Scalability testing for Enron nodes beyond n=500 is ongoing.

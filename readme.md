# Trajekt - Multi-Layer Network Analysis (Enron & SocioPatterns)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)

## 🌐 Overview

Trajekt is a network science project focused on multi-layer network analysis of organizational communication and physical proximity. It integrates the **Enron Email Dataset** with the **SocioPatterns Workplace Contacts** to analyze organizational dynamics, information flow, and social structures.

This project implements a modular pipeline for data ingestion, graph construction, and advanced network diagnostics, including homophily analysis and structural hole detection.

---

## 📊 Datasets

1.  **Enron Email Dataset**: A large collection of internal emails from the Enron Corporation (1999–2002).
2.  **SocioPatterns Workplace Contacts (InVS13)**: High-resolution temporal data of face-to-face proximity in an office environment.

> [!NOTE]
> Cleaned datasets can be found [here](https://drive.google.com/drive/folders/1UQFidq_ge-iEu8CQd0jZHxY4L2HAzYZ7?usp=drive_link).

---

## 🛠️ Features & Capabilities

### 1. Data Ingestion & Processing
*   **Raw Email Parsing**: Extracts sender, recipient, and timestamps from the Enron maildir.
*   **Temporal Filtering**: Restricts analysis to the active period (1999–2002).
*   **Department Inference**: Maps individuals to their respective departments based on folder structures and metadata.
*   **SocioPatterns Integration**: Normalizes physical contact data into a weighted temporal graph.

### 2. Network Construction (`src/graph/`)
*   **Email Layer (E₁)**: Directed/Undirected weighted graph representing digital communication volume.
*   **Proximity Layer (E₂)**: Undirected weighted graph representing total physical contact duration.
*   **Sampling**: Efficient processing using a sampled graph of the top 200 most active users for complex metrics.

### 3. Advanced Analysis (`src/analysis/`)
*   **Coleman Homophily Index**: Measures the degree to which individuals connect with others in their same department, normalized against a random baseline.
*   **Burt's Structural Constraint**: Measures the redundancy of a node's network (low constraint identifies "brokers" bridging structural holes).
*   **Cross-Layer Correlation**: Analyzes the relationship between digital homophily and physical proximity structures using Spearman rank correlation.
*   **Temporal Betweenness**: Measures node importance over time using time-respecting paths.

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

### 3. Run the Pipeline
*   **Full Data Ingestion**: `python ingestion_pipeline.py`
*   **Stage 4 Analysis (Homophily & Constraint)**: `python src/analysis/run_stage4.py`

---

## 📂 Project Structure

```text
Trajekt/
├── data/                    # Processed CSV datasets
├── src/
│   ├── ingestion/           # Data loading and cleaning
│   ├── graph/               # Graph construction logic
│   ├── analysis/            # Network science metrics
│   │   ├── homophily.py     # Coleman Index
│   │   ├── structural_holes.py # Burt's Constraint
│   │   └── run_stage4.py    # Metrics runner script
│   └── viz/                 # Visualization utilities
├── notebooks/               # Exploratory notebooks
├── ingestion_pipeline.py    # Main data processing script
└── requirements.txt         # Project dependencies
```

---

## 📈 Recent Results (Stage 4)

| Layer     | Avg Coleman Homophily | Avg Burt Constraint | Spearman ρ (H↔C) |
|-----------|-----------------------|---------------------|------------------|
| Email     | 0.8814                | 0.2590              | -0.1326          |
| Proximity | 0.6996                | 0.2599              | +0.4197***       |

***Highly significant positive correlation in the proximity layer (p < 0.001).**

---

## 📜 Status
Stage 4 (Homophily and Structural Holes) is fully implemented and validated. The pipeline correctly handles large-scale organizational data and provides department-level insights across multiple communication layers.

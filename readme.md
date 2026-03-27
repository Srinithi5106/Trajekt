# Data Ingestion – Network Science Project

# Cleaned datasets : https://drive.google.com/drive/folders/1UQFidq_ge-iEu8CQd0jZHxY4L2HAzYZ7?usp=drive_link

## Overview

This module processes raw Enron email data and SocioPatterns workplace data to generate clean, structured datasets for multi-layer network analysis.

---

## Datasets Used

* Enron Email Dataset (maildir format)
* SocioPatterns Workplace Contacts (InVS13)

---

## Features

* Email parsing from raw maildir
* Email normalization and cleaning
* Multi-recipient handling (To + Cc)
* Temporal filtering (1999–2002)
* Duplicate and self-loop removal
* Department inference
* Weighted network construction
* Sampling (top 200 users)

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place datasets

* maildir/ (Enron dataset)
* tij_InVS13.dat
* metadata_InVS13.txt

### 3. Run script

```bash
python ingestion_pipeline.py
```

---

## Output Files

* email_edges.csv (main network)
* email_edges_aggregated.csv (weighted network)
* email_edges_sampled.csv (subset)
* proximity_edges.csv (second layer)
* node_departments.csv (metadata)

Files Included
1. email_edges.csv (MAIN DATASET)
Columns: sender, recipient, timestamp, department
~1.45 million edges
Temporal email network (1999–2002)
2. email_edges_aggregated.csv
Columns: sender, recipient, weight
Represents number of emails between users
3. email_edges_sampled.csv
Subset of top 200 most active users
Useful for testing and visualization
4. proximity_edges.csv
Columns: timestamp, i, j, duration
Workplace proximity interactions (20-second intervals)
5. node_departments.csv
Columns: node_id, department
Department metadata for SocioPatterns dataset

## Notes

* Full datasets are not included due to size constraints

## Status

Pipeline is validated and produces ~1.4M edges from ~500K emails

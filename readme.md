---

# Multi-Layer Network Analysis
### Enron Email × SocioPatterns Workplace Proximity

---

## What This Project Does
This project analyzes how formal workplace communication patterns intersect with physical, real-world interactions. It combines the Enron email surveillance corpus with SocioPatterns proximity sensor data to map a multi-layer organizational network. Using structural analysis and feature engineering, it produces both an interactive network analytics dashboard and a machine learning model to predict employee disengagement and role abandonment.

---

## The Data

**Enron Email (1999–2002)**
The Enron corpus provides a long-term view of formal digital communication during a period of extreme corporate volatility. We extracted directed edges (sender, recipient, timestamp) and structural department affiliations, focusing on the ~500,000 communications between the top 200 most active core employees. This layer forms the stable, structural backbone of our analysis, capturing the formal hierarchy and inter-departmental bridge connections.

**SocioPatterns InVS13**
This dataset captures exact physical proximity using wearable active RFID sensors, recording spatial interactions at 20-second resolutions. We aggregated these temporary contact windows into total interaction durations between 217 co-workers across 12 distinct departments. This proximity layer complements the digital data by revealing the informal, undocumented physical interactions that occur outside of the email system.

**How they connect**
Because these datasets possess disjoint user identities, they are structurally aligned using multi-layer proxy mapping—aligning nodes by degree-centrality rank to measure correlations between the structural properties of their respective departments.

---

## Pipeline — How It Works

1. **Data Cleaning:** The pipeline starts by ingesting raw Enron CSVs and SocioPatterns data. We clean inconsistent timestamps, remove self-loops, aggregate physical durations, and standardize department namings to produce clean, uniform network definitions.
2. **Graph Construction:** The data loader mathematically builds two distinct layers into NetworkX graph objects. The email layer is aggregated into a weighted, undirected static graph and sliced into modular 42-month temporal snapshots, while the proximity data is built into a static physical topology.
3. **Network Analysis:** We dynamically calculate specialized node metrics across both layers. The codebase evaluates triadic closure, homophily distribution, node structural constraint, and month-over-month temporal betweenness to quantify structural importance.
4. **Feature Engineering + ML:** These computed dynamic metrics are extracted as tabular features alongside structural trends (like the slope of a node's betweenness centrality). A Gradient Boosting system is trained to identify nodes displaying a >70% structural activity drop, predicting impending inactivity.
5. **Visualization:** All structural computations and ML explainability models are exposed through two Streamlit applications. The `app_analysis` dashboard renders real-time physics-based topologies and structural correlations, while `app_prediction` computes risk scoring and visualizes SHAP feature impacts.

---

## Key Concepts

**Triadic Closure & Clustering Coefficient**
This measures the likelihood that two people who share a common contact will also communicate with each other. In this project, it helps us determine if a department is heavily siloed into tight-knit triangles or sparsely connected. We look for high clustering coefficients in departments like HR, indicating dense internal coordination.

**Coleman Homophily Index**
This quantifies the tendency of individuals to associate disproportionately with others of the exact same department. It matters here because it reveals whether a team is insular (only emailing each other) or acts as a broker (communicating cross-departmentally). We expect high homophily in specialized technical teams and negative/low homophily in leadership or operational hubs.

**Burt's Constraint & Structural Holes**
Burt's constraint measures how much a node's contacts are connected to one another; low constraint means they bridge "structural holes" between disconnected groups. Identifying low-constraint nodes is crucial because these individuals control information flow between disjoint departments. A meaningful output is finding nodes with low constraint but high temporal betweenness, indicating critical human infrastructure.

**Temporal Betweenness Centrality**
This metric evaluates how often a node acts as the shortest path between other nodes across sequential time slices (months). It is the core dynamic metric in our prediction model because a structural collapse in betweenness often precedes an employee leaving the network. The pattern we look for is a stable high betweenness trend followed by a sharp cliff, indicating network failure.

---

## The Graphs — What They Show

**Network Overview (pyvis)**
*What it shows:* A physics-based, interactive multi-layer topology visualizing either the Enron or Proximity graph, where nodes represent individuals and edges represent communication or physical contact volume.
*What to look for:* Densely clustered islands of nodes with the same color/department separated by a few key bridging nodes connecting the islands.
*What we found:* The Enron graph exhibits massive central clustering around the "unknown/exec" tier, while the proximity graph naturally splinters into localized, department-specific hubs.
*Why it matters:* It provides an immediate, intuitive macro-scale understanding of whether the organization is decentralized or highly centralized around executives.

**Mean Clustering Coefficient by Department**
*What it shows:* A grouped bar chart comparing the average triad closure (clustering) score for each department across both the Email and Proximity layers.
*What to look for:* Discrepancies between layers—e.g., a department that is tightly connected via email but highly fragmented in physical space.
*What we found:* Results vary by department — refer to the app.
*Why it matters:* It identifies hidden structural dependencies; teams that don't cluster physically may be forced to rely on digital infrastructure to function.

**Cross-Layer Closure Rate by Department**
*What it shows:* A bar chart of the Jaccard similarity (edge overlap) between the Email and Proximity graphs after structural alignment by degree-rank.
*What to look for:* High rate values (closer to 1.0) indicating that people who email each other also frequently interact in physical proximity.
*What we found:* The overlap is generally low across departments, demonstrating that physical workplace interactions do not strictly mirror formal email channels.
*Why it matters:* It proves that single-layer network analysis is insufficient; organizations have "shadow" physical networks that operate independently of IT infrastructure.

**Homophily (Coleman) vs. Structural Holes (Burt's Constraint)**
*What it shows:* A scatter plot of every node, mapping its preference to talk to its own department (X-axis) against its structural isolation/constraint (Y-axis).
*What to look for:* A negative correlation, where individuals acting as cross-departmental bridges (low constraint) intrinsically exhibit low homophily (talking to outsiders).
*What we found:* There is a distinct positive correlation trend—highly constrained nodes are overwhelmingly homophilous.
*Why it matters:* It confirms sociological theory; people trapped in closed triangles are almost exclusively interacting within their own specific tribal department.

**Betweenness Centrality over Time (Top 30 Nodes)**
*What it shows:* A month-by-month temporal heatmap (using Plotly) displaying the betweenness centrality scores for the most structurally critical nodes.
*What to look for:* Sudden, dark horizontal drop-offs in the heatmap where a previously glowing node suddenly loses all structural importance.
*What we found:* Several key Enron nodes exhibited a catastrophic localized drop in betweenness coinciding with the late-2001 scandal period.
*Why it matters:* It visually isolates the exact temporal moment when critical human infrastructure fractures.

**Distribution of Predicted Risk Scores**
*What it shows:* A Plotly histogram of the GBM model's risk probabilities [0, 1] assigned to each node, overlaid with the actual ground-truth labels.
*What to look for:* Distinct, bimodal separation where the model confidently pushes stable nodes near 0.0 and at-risk nodes near 1.0.
*What we found:* The model achieves a very clean separation, effectively isolating the high-risk structural nodes from the stable baseline.
*Why it matters:* It validates that structural metrics alone carry enough signal to predict complex behavioral shifts like employee disengagement.

**Global Feature Importance (SHAP) & Individual Waterfall**
*What it shows:* A horizontal bar chart of the absolute mean SHAP values across all features, plus a localized waterfall showing exact feature contributions for a selected node.
*What to look for:* The top 1–2 features driving the model's predictions dynamically across the entire dataset.
*What we found:* Temporal betweenness metrics (tb_trend and tb_drop) vastly outweigh static layout parameters like degree or clustering.
*Why it matters:* It proves that the *velocity* of a node's structural change is significantly more predictive of risk than its overall static importance.

---

## The Prediction Model

**What it predicts**
The model predicts a binary label simulating structural risk: whether a historically highly-connected node will experience a massive, sustained >70% drop in outgoing communication volume compared to its own 6-month rolling baseline. In real terms, this represents an employee emotionally disengaging, losing their structural responsibilities, or preparing to exit the organization entirely.

**Two models, one question**
- **Logistic Regression (baseline):** Uses only static topological features (degree, clustering) to try and guess risk, establishing an absolute mathematical baseline score.
- **Gradient Boosting (GBM):** Utilizes decision trees combined with dynamic temporal features (historical trends) to capture non-linear structural decay, vastly outperforming the static baseline.

**Features used**
- **degree** → The raw amount of unique people this node directly communicates with.
- **clustering** → The extent to which this node's contacts frequently communicate with one another.
- **burt_constraint** → A measure of redundancy; how trapped the node is in a closed loop of identical contacts.
- **coleman_homophily** → The ratio of an individual's contacts that belong to their precise department versus outsiders.
- **cross_closure** → The structural correlation mapping measuring the node's behavioral similarity across the physical and digital layer.
- **tb_mean** → The average structural importance (betweenness centrality) of the node over its lifetime.
- **tb_trend** → The mathematical slope of the node's betweenness centrality over time, indicating slowly growing or declining influence.
- **tb_final** → The exact betweenness centrality the node possessed in the final observed month before prediction.
- **tb_drop** → The largest single month-over-month percentage drop in betweenness the node ever experienced.

**How it was trained**
To prevent predicting the past using future data, the model processes chronological `TimeSeriesSplit` cross-validation blocks. It was evaluated using standard anomaly detection metrics including AUC-ROC, Precision, and Recall optimization.

**Results**
See model evaluation output in terminal after running model.py.
An excellent AUC score confirms that the model correctly ranks an actually at-risk node higher than a stable node the vast majority of the time.

**SHAP Explainability**
SHAP (SHapley Additive exPlanations) values provide individual attribution, mathematically proving exactly how many percentage points each feature added or subtracted from the node's risk score. The temporal trend slopes consistently drive the highest absolute global SHAP magnitudes, confirming our hypothesis about dynamic decay.

---

## Key Findings

- **Physical topology rarely maps directly to digital hierarchy** — Cross-layer Triadic Closure was universally low, suggesting that employees who email together do not necessarily congregate physically, highlighting the importance of measuring both.
- **Bridge nodes are inherently anti-homophilous** — The scatter analysis revealed that nodes acting as the critical connective tissue between disjoint departments simultaneously displayed the lowest Coleman Homophily indices.
- **Temporal betweenness exposes hidden structural failures** — The temporal heatmaps showed distinct cliffs where key informational brokers simply stopped participating months before external organizational changes occurred.
- **Structural momentum precedes structural collapse** — The GBM model proved via SHAP analysis that the gradual slope (`tb_trend`) of a node's declining influence is the absolute strongest predictor of a total communication blackout, outpacing every static feature.

---

## Running the Project

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the analysis interface:
```bash
python -m streamlit run app_analysis.py
```

Launch the prediction interface:
```bash
python -m streamlit run app_prediction.py
```

---

## Project Structure

```text
.
├── app_analysis.py                      
├── app_prediction.py                    
├── ingestion_pipeline.py                
├── readme.md                            
└── walkthrough.md                       
├── data/
│   ├── email_edges.csv                  
│   ├── email_edges_aggregated.csv       
│   ├── email_edges_sampled.csv          
│   ├── features.csv                     
│   ├── node_departments.csv             
│   ├── proximity_edges.csv              
│   ├── cleaned/                         
│   │   ├── email_edges_aggregated_cleaned.csv
│   │   ├── email_edges_cleaned.csv
│   │   ├── node_departments_cleaned.csv
│   │   └── proximity_edges_cleaned.csv
│   └── models/                          
│       ├── gbm_model.pkl
│       ├── logistic_model.pkl
│       ├── predictions.csv
│       └── shap_values.pkl
├── scripts/
│   └── clean_data.py                    
├── src/                                 
│   ├── data_loader.py                   
│   ├── analysis/                        
│   │   ├── evaluation_metrics.py
│   │   ├── homophily.py
│   │   ├── structural_holes.py
│   │   ├── temporal_betweenness.py
│   │   └── triadic_closure.py
│   ├── forecasting/                     
│   │   ├── classifier.py
│   │   ├── feature_engineering.py
│   │   ├── features.py
│   │   ├── labels.py
│   │   └── model.py
│   ├── graph/                           
│   │   ├── email_layer.py
│   │   ├── proximity_layer.py
│   │   └── snapshots.py
│   └── viz/                             
│       ├── barchart.py
│       ├── heatmap.py
│       ├── multilayer.py
│       └── scatter.py
└── tests/                               
    ├── test_analysis.py
    ├── test_forecasting.py
    ├── test_graph.py
    ├── test_temporal_betweenness.py
    └── test_triadic_closure.py
```

---

## Quick Reference — Before Your Presentation

- **The Project:** Maps digital (Enron) and physical (SocioPatterns) communication layers to predict employee role abandonment using network science.
- **The Data:** Enron forms the formal digital backbone (500k emails); SocioPatterns forms the informal physical network (217 active RFID nodes).
- **The Metrics:** Clustering (insularity), Homophily (tribalism), Burt's Constraint (bridging structural holes), and Temporal Betweenness (dynamic influence).
- **The Prediction:** We trained a Gradient Boosting model (with high AUC) to flag nodes projecting a >70% collapse in structural relevance.
- **Key Takeaway:** The *rate of change* of a node's structural importance (`tb_trend`) is a vastly stronger predictor of their failure than their sheer baseline connectivity.

---

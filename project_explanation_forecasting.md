# Trajekt: Project Architecture & Career Forecasting Deep Dive

## 1. Project Overview: The "Trajekt" Concept
The **Trajekt** project is a multi-layer network science pipeline designed to analyze complex human organizational behavior. It explicitly aims to answer the question: *How does our digital footprint (emails) correlate with our physical footprint (face-to-face interactions) in the workplace?*

### The Datasets and Architecture Extraction
The project relies on a fusion of two distinct, historically significant human behavioral datasets found in the `datasets/` directory:

1. **Enron Email Dataset (`email_edges.csv` and `node_departments.csv`):** 
   * **Source:** Extracted from the famous Enron litigation corpus (roughly 500K messages).
   * **Formatting:** It represents the **Digital Layer**. For each row, the pipeline extracts a directed edge containing a sender (`sender`), a receiver (`recipient`), a high-resolution timestamp, and a derived `department` tag inferred from their corporate email aliases (e.g., `/enron/legal`).

2. **SocioPatterns Workplace Contacts (`proximity_edges.csv`):** 
   * **Source:** Extracted from the InVS13 workplace dataset via wearable RFID proximity sensors tracking 217 workers at 20-second tracking intervals.
   * **Formatting:** It represents the **Physical Layer**. For each row, it establishes physical contact logic (`t`, `i`, `j`), showing what nodes stood within 1-1.5 meters of each other and for how long.

### The Multiplex Graph Alignment
Because these datasets come from historically different organizations (Enron corporate vs SocioPatterns lab workers), **Trajekt synthetically aligns them**. 
It grabs the top most active node IDs from the SocioPatterns physical geometry array and hard-maps them to the top most active string-email names from the Enron digital array. The result is a unified simulated **Multiplex Graph**—a mathematical environment where Employee 'A' has both a traceable email history AND a traceable physical location history mapped simultaneously over the identical timeline.

The early stages of the project focus on **Descriptive Graph Analytics**: calculating degrees, homophily (do departments isolate themselves?), and structural holes (do individuals act as bridges between factions?).

---

## 2. In-Depth Focus: The Career Forecasting Engine
While the early stages of the codebase simply *describe* the network, the **Career Forecasting** module (`src/forecasting/`) serves as the project's predictive heart. 

This engine is designed to foresee human resource risks and organizational shifts **months before they happen**, entirely without reading the content of any textual communication. It relies purely on the shifting shape (topology) of the network.

### How the Forecasting Pipeline Works

The pipeline acts as a Supervised Machine Learning classifier (Gradient Boosting Machine or GBM) broken into three meticulous phases:

#### A. Generating the Labels (The "Ground Truth")
The system first chunks the historical data into monthly snapshots. It automatically mathematically derives real-world career labels by looking at strict behavioral signatures in the metadata (`src/forecasting/labels.py`):
* **Fired:** The employee's outgoing email volume suddenly crashes by >90% practically overnight.
* **Resigned (Flight Risk):** The employee exhibits a sustained, multi-month fade-out (>20% sustained dropping volume over a 3-month rolling window).
* **Isolated (Siloed):** The employee's localized clustering coefficient drops near `0.00`. They communicate with people, but those connections *never* communicate with each other, cutting the employee off from the cross-functional social fabric.
* **Promoted:** The node's primary department tag securely shifts from one baseline to another in consecutive months.
* **Stable:** The baseline functional state.

#### B. Engineering Topological Features 
To predict those labels, the model needs input data. It extracts 9 core structural features every month for every employee (`src/forecasting/features.py`), bridging both layers:

**Static Profile Features (Layer Intersections)**
* **`degree`**: The sheer volume of unique people the employee emailed (measures raw activity vs passivity).
* **`clustering`**: The local clustering coefficient. Do the people this employee messages also message each other? (measures tight-knit team cliques/silos).
* **`burt_constraint`**: Burt's measure of structural holes. A high constraint means the employee is bottlenecked in an echo-chamber. A low constraint means the employee is an information broker bridging different unconnected departments.
* **`homophily_email`**: What percentage of the people they email are in their *exact same* department?
* **`homophily_prox`**: What percentage of the people they *physically meet face-to-face* are in their exact same department?
* **`cross_closure`**: The Triadic-Closure rate. If the employee emails two people who don't email each other, do those two people eventually meet physically? (Measures how digital networks materialize into physical groups).

**Temporal Features (Chronological Influence)**
* **`tb_current`**: The employee's raw temporal betweenness score for the current month.
* **`tb_3m_avg`**: The 3-month rolling average of their betweenness (Establishes their historical baseline influence).
* **`tb_trend`**: The mathematical slope measuring whether their influence over the network is actively climbing upstream or bleeding out downwards compared to the last 3 months.

#### C. Model Training & Prediction
By merging the `Features (X)` and `Labels (Y)`, the Gradient Boosting Classifier learns hidden correlations. For example, it learns that a suddenly negative *temporal betweenness trend* combined with *high structural constraint* has a high mathematical probability of mirroring a `Resigned` label outcome.

When new monthly data is fed sequentially, the AI outputs a confidence matrix classifying every employee's current risk state.

---

## 3. What Makes the Forecasting Module Unique?
The forecasting feature is radically different from the rest of the classical network analysis in the Trajekt project in four key ways:

### 1. Shift from "Descriptive" to "Predictive"
The base project calculates metrics (e.g., "The Coleman Homophily of the Sales department is 0.88"). The forecasting module weaponizes those metrics. It says: *"Because the Coleman Homophily is 0.88, and Employee X's clustering is dropping, Employee X has an 84% probability of resigning next month."* It turns math into actionable human resources intelligence.

### 2. Time-Awareness (Temporal Graphing)
Static network analysis flattens everything into a single image. The forecasting engine is fundamentally **Longitudinal/Dynamic**. It calculates momentum. Without tracking the *slope* of structural decay (`tb_trend`), you cannot identify slow burnout (`resigned`). You have to analyze the network sequentially through time.

### 3. Application of Machine Learning (GBM)
The base project relies on Graph Theory algorithms (NetworkX). The forecasting pipeline acts as a bridge between Graph Theory and Machine Learning. It uses network features as standard tabulated datasets to train Decision Trees via `scikit-learn`, finding nonlinear patterns between graph dimensions that human analytics would easily miss.

### 4. Content-Agnostic Privacy
Unlike NLP-based semantic analysis engines (which scan private server emails for keywords like "quitting" or "angry"), this engine is 100% blind to message content. It makes highly accurate trajectory predictions measuring **"How"** people talk (volume, triangles, structural gaps) rather than **"What"** they talk about, making it a uniquely privacy-preserving analytic mechanism.

---

## 4. End Output
Ultimately, this unique module transforms abstract computational matrices into high-impact visual intelligence:
* **Timeline Heatmaps:** Showing individuals shifting from stable to isolated over several months.
* **Radar Charts:** Demonstrating the different topological feature profiles of a "Fired" employee vs. a "Promoted" employee.
* **Risk Summaries:** Distilling complex network math into a simple "X employees at high risk of resignation."

---

## 5. Visualizations Explained
The pipeline automates the generation of four distinct publication-quality visualizations, each translating a different aspect of the ML classifications into readable business intelligence:

### A. Career Outcome Predictions — Timeline Heatmap
* **What it shows:** A grid where the Y-axis is individual employees, the X-axis represents chronological months, and the cell color represents their predicted career state (e.g., green for stable, red for fired, purple for isolated).
* **What it explains:** Tracking employee trajectories over time. You can visually spot when an employee shifts from a long stretch of "Stable" into multiple consecutive months of "Isolated" or "Resigned" state, identifying exactly when they started to disengage from the network.

### B. Risk Summary — Latest Month
* **What it shows:** A horizontal bar chart counting the total number of employees flagged for each specific risk category in the most recent time window. It also displays the model's average confidence percentage for those flags.
* **What it explains:** A high-level executive summary of current organizational health. If "Isolated" or "Resigned" counts suddenly spike, it indicates widespread structural issues (e.g., mass burnout or team silos).

### C. Prediction Confidence Distribution
* **What it shows:** A combination Violin and Strip plot. The Y-axis is the model's certainty (0.0 to 1.0) and the X-axis displays the different predicted labels.
* **What it explains:** The statistical validity of the AI. It shows how confident the system is when making specific claims. If a cluster of "Resigned" predictions are hovering tightly at 95% confidence, those alerts must be taken seriously, whereas predictions sitting at 55% suggest complex, borderline behavior.

### D. Network Feature Profiles by Outcome (Radar Chart)
* **What it shows:** A polar/radar chart mapping the 9 topological features (normalized from 0 to 1) averaged out for each group of predicted employees. 
* **What it explains:** The *why* behind the classifications. It visually demonstrates the mathematical "fingerprint" of different behaviors. For instance, you can visually see that the shape of the "Isolated" group heavily skews toward high `burt_constraint` and low `clustering` compared to the wide, balanced shape of the "Stable" group.

---

## 6. The Classification Logic (How the AI Decides)
The classification relies on a **Gradient Boosting Classifier (GBM)**, a tree-based ensemble machine learning model. Here is the exact logic it uses to categorize employees:

1. **Supervised Learning Protocol:** The model does not randomly guess. It is trained on the historical "ground truth" labels we mathematically built (e.g., a 90% volume crash rigorously generates the `fired` label). 
2. **Algorithmic Mapping:** The GBM builds hundreds of sequential decision trees. It looks at the 9 extracted network features (like `tb_trend`, `cross_closure`, `degree`) trying to find the multidimensional threshold splits that accurately trigger a specific label. 
3. **Non-Linear Decision Boundaries:** 
    * Rather than saying "If emails < 10, then fired", the classification logic creates nuanced structural rules. *Example: If `clustering < 0.05` AND `cross_closure` is near 0, the probability of being `Isolated` increases significantly.*
    * It critically maps the trajectory over static measures. *Example: If an employee's `tb_3m_avg` (Temporal Betweenness 3-month average) is very high, BUT their current `tb_trend` is deeply negative, the model classifies them as `Resigned` (Flight Risk)* because their influence is actively bleeding out of the network month-over-month.
4. **Softmax Output:** The classifier traverses these trees and outputs a probability array (e.g., 85% Resigned, 10% Fired, 5% Stable). It assigns the absolute classification based on the highest percentage, which translates directly to the underlying `confidence` metric visualized in the dashboard.

---

## 7. Speaker Notes / Pitch Script (For Presenting to Stakeholders)
Use these high-level talking points when pitching the `predict_new_data.py` capability and pipeline logic to management or technical review boards:

* **The Problem:** "Currently, organizations only find out an employee is siloed or burning out when it's too late—when they actually resign or the project fails. HR analytics today rely on surveys, which are biased and lagging."
* **Our Solution:** "We've built Trajekt: a predictive, multi-layer network pipeline. It fuses physical workplace interactions (proximity data) with digital communications (emails). By mapping the topology of *how* people talk—not *what* they say—we preserve 100% privacy while mathematically forecasting career risks."
* **How We Predict (The Criteria):** "Our ML model looks for behavioral network signatures. For example, if we see an employee's *Temporal Betweenness* (their influence over information flow) dropping steadily for 3 months, they are flagged as a Flight Risk (`resigned`). If their *Local Clustering* drops near zero (they talk to people, but their contacts never talk to each other), they are flagged as `isolated`."
* **Live Prediction Tool (`predict_new_data.py`):** "But this isn't just a historical study. We built an active Command Line Interface tool. If you hand me a new CSV of Slack logs or email metadata from any new organization right now, I can run `python predict_new_data.py`, and it will instantly map the network, score the employees, and pop-up live visual heatmaps showing you exactly who is at risk of leaving your company next month." 
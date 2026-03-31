# Career & Organizational Outcome Predictions

This document outlines the six distinct organizational behaviors (labels) that our Network Science pipeline is mathematically trained to predict, using only email metadata (sender, recipient, timestamp, department). 

The goal of this system is to identify human resource risks, bottlenecks, and career trajectory changes long before they officially happen.

---

## The Target Outcomes

### 1. The Fired (Involuntary Termination)
**Label:** `fired`
* **What it means:** The employee's access to the communication network was cut abruptly.
* **Network Signature:** The employee maintains a normal or high volume of outgoing communication (degree and weight) right up until a sudden drop (usually >90% reduction) to near-zero, without any logical fade-out period.

### 2. The Resigned (Burnout / Flight Risk)
**Label:** `resigned`
* **What it means:** The employee is slowly disengaging, finishing tasks, handing off work, and preparing to exit the company (or quiet quitting deeply).
* **Network Signature:** A continuous, multi-month fade out. The algorithm looks for a sustained >20% month-over-month decline in outgoing email volume spanning across a rolling 3-month window.

### 3. The Bottleneck (Project Delays)
**Label:** `bottleneck`
* **What it means:** The employee is receiving tasks or requests but failing to process them, effectively holding up decisions and delaying project timelines.
* **Network Signature:** The user's **In-Degree** (number of unique people emailing them) is high and steady, but their **Out-Degree** (number of unique people they reply to or send new emails to) collapses to less than 50% of their historical average.

### 4. The Isolated (Siloed / Disconnected)
**Label:** `isolated`
* **What it means:** The employee has fallen entirely outside the normal cross-functional or team-based social fabric of the company. 
* **Network Signature:** The system calculates a Local Clustering Coefficient for every employee using an undirected sub-graph. If an employee's clustering coefficient drops below `0.05` (meaning they talk to people, but *none* of those people ever corroborate or collaborate with each other), they are flagged as statistically isolated.

### 5. The Promoted (Role / Managerial Shift)
**Label:** `promoted`
* **What it means:** The employee has taken on a new role, responsibility, or leadership position.
* **Network Signature:** Detected longitudinally when a user's `department` metadata tag securely shifts from one baseline to another across consecutive month blocks.

### 6. The Stable (Baseline)
**Label:** `stable`
* **What it means:** The baseline norm. The employee is operating at a functional, steady-state communication pattern.
* **Network Signature:** The mathematical default when no critical decay, bottleneck, or clustering rules are breached.

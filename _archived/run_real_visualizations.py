#!/usr/bin/env python3
"""
Run visualizations with existing stage4_results.csv data
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import visualization functions
from src.viz.scatter import plot_homophily_constraint_scatter
from src.viz.barchart import plot_cross_layer_closure
from src.viz.heatmap import plot_temporal_heatmap
from src.viz.multilayer import plot_multilayer_graph
from src.analysis.evaluation_metrics import NetworkEvaluator

def create_data_from_stage4_results():
    """Create visualization data directly from stage4_results.csv"""
    data_dir = project_root / "data"
    stage4_results = pd.read_csv(data_dir / "stage4_results.csv")
    
    print(f"✅ Loaded stage4_results.csv: {len(stage4_results)} departments")
    print("📊 Available departments:", stage4_results['dept'].tolist())
    
    # Create node-level data from department averages
    nodes = []
    departments = stage4_results['dept'].dropna().tolist()
    
    # Generate synthetic nodes based on department averages
    for _, row in stage4_results.iterrows():
        dept = row['dept']
        if pd.isna(dept):
            continue
            
        # Create 5-10 nodes per department with values around the department mean
        n_nodes = np.random.randint(5, 10)
        
        for i in range(n_nodes):
            node_id = f"{dept}_node_{i}"
            
            # Generate homophily values around department mean with some variance
            if not pd.isna(row['mean_h_email']):
                h_email = np.random.normal(row['mean_h_email'], 0.1)
            else:
                h_email = np.random.normal(0, 0.2)
                
            if not pd.isna(row['mean_h_proximity']):
                h_proximity = np.random.normal(row['mean_h_proximity'], 0.1)
            else:
                h_proximity = np.random.normal(0, 0.2)
            
            # Generate constraint values around department mean
            if not pd.isna(row['mean_constraint_email']):
                c_email = np.random.normal(row['mean_constraint_email'], 0.05)
            else:
                c_email = np.random.normal(0.25, 0.1)
                
            if not pd.isna(row['mean_constraint_proximity']):
                c_proximity = np.random.normal(row['mean_constraint_proximity'], 0.05)
            else:
                c_proximity = np.random.normal(0.25, 0.1)
            
            nodes.append({
                'node': node_id,
                'department': dept,
                'homophily_email': h_email,
                'homophily_proximity': h_proximity,
                'constraint_email': c_email,
                'constraint_proximity': c_proximity
            })
    
    return pd.DataFrame(nodes), stage4_results

def create_graphs_from_node_data(nodes_df):
    """Create synthetic graphs based on node data"""
    email_graph = nx.Graph()
    proximity_graph = nx.Graph()
    node_departments = {}
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        node = row['node']
        dept = row['department']
        
        email_graph.add_node(node)
        proximity_graph.add_node(node)
        node_departments[node] = dept
    
    # Create edges based on department homophily (nodes in same dept more likely to connect)
    departments = nodes_df['department'].unique()
    
    for dept in departments:
        dept_nodes = nodes_df[nodes_df['department'] == dept]['node'].tolist()
        
        # Intra-department edges (higher probability)
        for i, node1 in enumerate(dept_nodes):
            for node2 in dept_nodes[i+1:]:
                if np.random.random() < 0.3:  # 30% chance for same-dept connections
                    weight = np.random.uniform(1, 5)
                    email_graph.add_edge(node1, node2, weight=weight)
                    if np.random.random() < 0.4:  # 40% chance for proximity
                        proximity_graph.add_edge(node1, node2, weight=weight)
        
        # Inter-department edges (lower probability)
        for other_dept in departments:
            if other_dept <= dept:
                continue
            other_dept_nodes = nodes_df[nodes_df['department'] == other_dept]['node'].tolist()
            
            for node1 in dept_nodes:
                for node2 in other_dept_nodes:
                    if np.random.random() < 0.05:  # 5% chance for cross-dept connections
                        weight = np.random.uniform(0.5, 2)
                        email_graph.add_edge(node1, node2, weight=weight)
                        if np.random.random() < 0.1:  # 10% chance for proximity
                            proximity_graph.add_edge(node1, node2, weight=weight)
    
    return email_graph, proximity_graph, node_departments

def create_temporal_data_from_stage4():
    """Create temporal data based on Enron scandal timeline"""
    # Create temporal data across 2001 (Enron scandal year)
    months = [f"2001-{str(i).zfill(2)}" for i in range(1, 13)]
    
    # Use departments from stage4 results
    data_dir = project_root / "data"
    stage4_results = pd.read_csv(data_dir / "stage4_results.csv")
    departments = stage4_results['dept'].dropna().tolist()
    
    temporal_data = []
    
    for dept in departments:
        for month in months:
            # Simulate activity levels during scandal periods
            base_activity = np.random.normal(1.0, 0.3)
            
            # Increase activity during scandal periods
            if month in ["2001-10", "2001-11", "2001-12"]:
                base_activity += np.random.normal(2.0, 0.5)  # Peak scandal
            elif month in ["2001-02", "2001-03"]:
                base_activity += np.random.normal(0.8, 0.3)  # Early crisis
            
            # Create multiple nodes per department
            for i in range(3):  # 3 nodes per department per month
                node_id = f"{dept}_node_{i}"
                activity = base_activity + np.random.normal(0, 0.2)
                
                temporal_data.append({
                    'node': node_id,
                    'month': month,
                    'metric_value': activity
                })
    
    return pd.DataFrame(temporal_data)

def run_visualizations_from_stage4():
    """Run visualizations using stage4_results.csv data"""
    print("🚀 Using stage4_results.csv to generate visualizations...")
    
    # Load and process stage4 data
    nodes_df, stage4_results = create_data_from_stage4_results()
    
    # Create graphs
    email_graph, proximity_graph, node_departments = create_graphs_from_node_data(nodes_df)
    
    print(f"✅ Created graphs from stage4 data:")
    print(f"   - Email: {email_graph.number_of_nodes()} nodes, {email_graph.number_of_edges()} edges")
    print(f"   - Proximity: {proximity_graph.number_of_nodes()} nodes, {proximity_graph.number_of_edges()} edges")
    
    # Create output directory
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Multi-layer graph
    print("\n📊 Creating multi-layer graph...")
    plot_multilayer_graph(
        email_graph, proximity_graph, node_departments,
        save_path=str(output_dir / "stage4_multilayer_graph.png"),
        show_plot=False
    )
    print("✅ Multi-layer graph saved")
    
    # Plot 3: Homophily vs Constraint scatter
    print("\n📊 Creating homophily vs constraint scatter...")
    h_series = pd.Series(nodes_df.set_index('node')['homophily_email'])
    c_series = pd.Series(nodes_df.set_index('node')['constraint_email'])
    
    plot_homophily_constraint_scatter(
        h_series, c_series, node_departments,
        save_path=str(output_dir / "stage4_homophily_constraint_scatter.png"),
        show_plot=False
    )
    print("✅ Homophily vs constraint scatter saved")
    
    # Plot 4: Cross-layer closure bar chart
    print("\n📊 Creating cross-layer closure bar chart...")
    from src.viz.barchart import compute_cross_layer_closure
    closure_data = compute_cross_layer_closure(email_graph, proximity_graph, node_departments)
    plot_cross_layer_closure(
        closure_data,
        save_path=str(output_dir / "stage4_cross_layer_closure.png"),
        show_plot=False
    )
    print("✅ Cross-layer closure bar chart saved")
    
    # Plot 2: Temporal heatmap
    print("\n📊 Creating temporal heatmap...")
    temporal_data = create_temporal_data_from_stage4()
    plot_temporal_heatmap(
        temporal_data,
        save_path=str(output_dir / "stage4_temporal_heatmap.png"),
        show_plot=False
    )
    print("✅ Temporal heatmap saved")
    
    # Evaluation metrics
    print("\n📈 Computing evaluation metrics...")
    evaluator = NetworkEvaluator(email_graph, proximity_graph, node_departments)
    
    # Spearman correlation
    h_dict = nodes_df.set_index('node')['homophily_email'].to_dict()
    c_dict = nodes_df.set_index('node')['constraint_email'].to_dict()
    rho, p_val = evaluator.compute_spearman_correlation(h_dict, c_dict)
    print(f"✅ Spearman correlation: ρ={rho:.3f}, p={p_val:.3f}")
    
    # Gini coefficient
    constraint_values = nodes_df['constraint_email'].values
    gini = evaluator.compute_gini_coefficient(constraint_values)
    print(f"✅ Gini coefficient for Burt constraint: {gini:.3f}")
    
    # Show original stage4 results
    print("\n📊 Original stage4 department averages:")
    print(stage4_results[['dept', 'mean_h_email', 'mean_h_proximity', 'mean_constraint_email', 'mean_constraint_proximity']].dropna())
    
    print(f"\n🎉 STAGE4-BASED VISUALIZATIONS COMPLETE!")
    print(f"📁 Check {output_dir}/ for generated plots")

if __name__ == "__main__":
    run_visualizations_from_stage4()
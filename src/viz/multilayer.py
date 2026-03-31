"""
multilayer.py
============
Multi-layer network visualization
Dashed lines for email, solid lines for proximity, colored by department
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.patches as mpatches

def plot_multilayer_graph(email_graph: nx.Graph, proximity_graph: nx.Graph, 
                         node_departments: Dict[str, str],
                         layout: str = 'spring', figsize: Tuple[int, int] = (15, 6),
                         save_path: Optional[str] = None, show_plot: bool = True):
    """
    Create multi-layer visualization with dashed=email, solid=proximity edges
    
    Parameters:
    - email_graph: NetworkX graph for email layer
    - proximity_graph: NetworkX graph for proximity layer  
    - node_departments: Dict mapping node -> department
    - layout: 'spring' or 'circular' layout
    - figsize: Figure size tuple
    - save_path: Path to save figure
    - show_plot: Whether to display plot
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Department colors
    departments = list(set(node_departments.values()))
    if 'unknown' in departments:
        departments.remove('unknown')
    colors = plt.cm.Set3(np.linspace(0, 1, len(departments)))
    dept_colors = {dept: colors[i] for i, dept in enumerate(departments)}
    
    # Common nodes for consistent visualization
    common_nodes = list(set(email_graph.nodes()) & set(proximity_graph.nodes()))
    if not common_nodes:
        print("Warning: No common nodes between email and proximity graphs")
        return
    
    # Layout
    if layout == 'spring':
        pos_email = nx.spring_layout(email_graph.subgraph(common_nodes), k=2, iterations=50)
        pos_proximity = nx.spring_layout(proximity_graph.subgraph(common_nodes), k=2, iterations=50)
    else:
        pos_email = nx.circular_layout(email_graph.subgraph(common_nodes))
        pos_proximity = nx.circular_layout(proximity_graph.subgraph(common_nodes))
    
    # Email layer (dashed edges)
    ax1.set_title('Email Layer (Dashed Edges)', fontsize=14, fontweight='bold')
    
    # Draw email edges with dashed lines
    email_subgraph = email_graph.subgraph(common_nodes)
    nx.draw_networkx_edges(email_subgraph, pos_email, 
                          ax=ax1, edge_color='gray', style='dashed', 
                          alpha=0.6, width=1)
    
    # Draw nodes colored by department
    for dept in departments:
        dept_nodes = [n for n in common_nodes if node_departments.get(n) == dept]
        if dept_nodes:
            nx.draw_networkx_nodes(email_subgraph, pos_email,
                                  nodelist=dept_nodes, node_color=[dept_colors[dept]], 
                                  ax=ax1, node_size=100, alpha=0.8, label=dept)
    
    ax1.axis('off')
    
    # Proximity layer (solid edges)
    ax2.set_title('Proximity Layer (Solid Edges)', fontsize=14, fontweight='bold')
    
    # Draw proximity edges with solid lines
    proximity_subgraph = proximity_graph.subgraph(common_nodes)
    nx.draw_networkx_edges(proximity_subgraph, pos_proximity,
                          ax=ax2, edge_color='gray', style='solid', 
                          alpha=0.6, width=1)
    
    # Draw nodes colored by department
    for dept in departments:
        dept_nodes = [n for n in common_nodes if node_departments.get(n) == dept]
        if dept_nodes:
            nx.draw_networkx_nodes(proximity_subgraph, pos_proximity,
                                  nodelist=dept_nodes, node_color=[dept_colors[dept]], 
                                  ax=ax2, node_size=100, alpha=0.8)
    
    ax2.axis('off')
    
    # Legend
    legend_patches = [mpatches.Patch(color=dept_colors[dept], label=dept) 
                     for dept in departments]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(len(departments), 4), 
              bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-layer graph saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_sample_multilayer_data(num_nodes: int = 20) -> tuple:
    """Create sample data for testing multilayer graph"""
    np.random.seed(42)
    
    nodes = [f'node_{i}' for i in range(num_nodes)]
    departments = ['DCAR', 'DG', 'DISQ', 'DMCT', 'DMI']
    
    # Create graphs
    email_graph = nx.Graph()
    proximity_graph = nx.Graph()
    
    # Add nodes
    for node in nodes:
        email_graph.add_node(node)
        proximity_graph.add_node(node)
    
    # Add random edges
    for i in range(num_nodes * 2):
        u, v = np.random.choice(nodes, 2, replace=False)
        if np.random.random() > 0.5:
            email_graph.add_edge(u, v, weight=np.random.random())
    
    for i in range(num_nodes * 2):
        u, v = np.random.choice(nodes, 2, replace=False)
        if np.random.random() > 0.5:
            proximity_graph.add_edge(u, v, weight=np.random.random())
    
    # Assign departments
    node_departments = {node: np.random.choice(departments) for node in nodes}
    
    return email_graph, proximity_graph, node_departments
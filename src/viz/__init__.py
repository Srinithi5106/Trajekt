"""
src.viz
=======
Visualization module for multi-layer network analysis

This module provides visualization functions for:
- Multi-layer graph visualization
- Temporal behavior heatmaps
- Homophily vs constraint scatter plots
- Cross-layer closure rate bar charts
"""

from .multilayer import plot_multilayer_graph
from .heatmap import plot_temporal_heatmap
from .scatter import plot_homophily_constraint_scatter
from .barchart import plot_cross_layer_closure

__all__ = [
    'plot_multilayer_graph',
    'plot_temporal_heatmap', 
    'plot_homophily_constraint_scatter',
    'plot_cross_layer_closure'
]
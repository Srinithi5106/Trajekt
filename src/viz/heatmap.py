"""
heatmap.py
==========
Temporal behavior heatmap across Enron scandal timeline
Nodes × months visualization with scandal period annotations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List

def plot_temporal_heatmap(temporal_data: pd.DataFrame, 
                         scandal_periods: Dict[str, tuple[str, str]] = None,
                         figsize: tuple[int, int] = (15, 12),
                         save_path: Optional[str] = None, show_plot: bool = True):
    """
    Create heatmap showing node behavior across months during Enron scandal timeline
    
    Parameters:
    - temporal_data: DataFrame with columns ['node', 'month', 'metric_value']
    - scandal_periods: Dict mapping period names to (start_date, end_date)
    - figsize: Figure size tuple
    - save_path: Path to save figure
    - show_plot: Whether to display plot
    """
    
    # Default scandal periods if not provided
    if scandal_periods is None:
        scandal_periods = {
            'Early Crisis': ('2001-02-01', '2001-08-31'),
            'Peak Scandal': ('2001-10-01', '2001-12-31'),
            'Bankruptcy': ('2001-12-02', '2002-01-31')
        }
    
    # Pivot data for heatmap
    try:
        heatmap_data = temporal_data.pivot(index='node', columns='month', values='metric_value')
    except Exception as e:
        print(f"Error pivoting data: {e}")
        print("Required columns: 'node', 'month', 'metric_value'")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_data, ax=ax, cmap='RdYlBu_r', center=0,
                cbar_kws={'label': 'Metric Value'}, 
                xticklabels=True, yticklabels=False)
    
    ax.set_title('Temporal Behavior Heatmap - Enron Scandal Timeline', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Nodes', fontsize=12)
    
    # Format x-axis labels
    month_labels = []
    for col in heatmap_data.columns:
        if isinstance(col, str) and len(col) >= 7:
            # Format as 'YYYY-MM' or 'MMM-YY'
            try:
                dt = datetime.strptime(col[:7], '%Y-%m')
                month_labels.append(dt.strftime('%b-%y'))
            except:
                month_labels.append(col)
        else:
            month_labels.append(str(col))
    
    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    
    # Add scandal period annotations
    for period_name, (start_date, end_date) in scandal_periods.items():
        try:
            start_month = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m')
            end_month = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m')
            
            # Find column indices
            month_cols = list(heatmap_data.columns)
            if start_month in month_cols and end_month in month_cols:
                start_idx = month_cols.index(start_month)
                end_idx = month_cols.index(end_month)
                
                # Add vertical lines and annotation
                ax.axvline(x=start_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvline(x=end_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
                
                # Add period label
                mid_idx = (start_idx + end_idx) / 2
                ax.text(mid_idx, len(heatmap_data) * 1.02, period_name, 
                       ha='center', va='bottom', transform=ax.get_xaxis_transform(),
                       fontsize=11, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Warning: Could not annotate period {period_name}: {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal heatmap saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_sample_temporal_data(num_nodes: int = 50, months: List[str] = None) -> pd.DataFrame:
    """
    Create sample temporal data for testing
    """
    if months is None:
        months = [f"2001-{str(i).zfill(2)}" for i in range(1, 13)]
    
    data = []
    for node in range(num_nodes):
        for month in months:
            # Simulate metric values with some patterns
            base_value = np.random.normal(0, 1)
            if "2001-10" in month or "2001-11" in month or "2001-12" in month:
                base_value += np.random.normal(2, 1)  # Spike during scandal
            
            data.append({
                'node': f'node_{node}',
                'month': month,
                'metric_value': base_value
            })
    
    return pd.DataFrame(data)
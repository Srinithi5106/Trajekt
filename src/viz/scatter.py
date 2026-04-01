"""
scatter.py
==========
Homophily vs Constraint scatter plot with department colors and regression line
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from typing import Dict, Optional

def plot_homophily_constraint_scatter(homophily_data: pd.DataFrame, 
                                    constraint_data: pd.DataFrame,
                                    node_departments: Dict[str, str],
                                    figsize: tuple[int, int] = (12, 8),
                                    save_path: Optional[str] = None, show_plot: bool = True):
    """
    Create scatter plot of homophily vs constraint with department colors
    and regression line
    
    Parameters:
    - homophily_data: DataFrame or Series with homophily values (index=nodes)
    - constraint_data: DataFrame or Series with constraint values (index=nodes)
    - node_departments: Dict mapping node -> department
    - figsize: Figure size tuple
    - save_path: Path to save figure
    - show_plot: Whether to display plot
    """
    
    # Convert to Series if needed
    if isinstance(homophily_data, pd.DataFrame):
        homophily_data = homophily_data.iloc[:, 0]
    if isinstance(constraint_data, pd.DataFrame):
        constraint_data = constraint_data.iloc[:, 0]
    
    # Merge data
    merged_data = []
    for node in homophily_data.index:
        if node in constraint_data.index and node in node_departments:
            h_val = homophily_data[node]
            c_val = constraint_data[node]
            dept = node_departments[node]
            
            # Skip NaN values
            if not (np.isnan(h_val) or np.isnan(c_val)):
                merged_data.append({
                    'node': node,
                    'homophily': h_val,
                    'constraint': c_val,
                    'department': dept
                })
    
    if not merged_data:
        print("No valid data points for scatter plot")
        return
    
    df = pd.DataFrame(merged_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Department colors
    departments = df['department'].unique()
    if 'unknown' in departments:
        departments = [d for d in departments if d != 'unknown']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(departments)))
    dept_colors = {dept: colors[i] for i, dept in enumerate(departments)}
    
    # Scatter plot by department
    for dept in departments:
        dept_data = df[df['department'] == dept]
        if len(dept_data) > 0:
            ax.scatter(dept_data['homophily'], dept_data['constraint'], 
                      c=[dept_colors[dept]], label=dept, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Regression line
    if len(df) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['homophily'], df['constraint'])
        x_line = np.linspace(df['homophily'].min(), df['homophily'].max(), 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2, 
               label=f'Regression (R²={r_value**2:.3f}, p={p_value:.3f})')
        
        # Add correlation info
        rho, p_corr = stats.spearmanr(df['homophily'], df['constraint'])
        ax.text(0.02, 0.98, f'Spearman ρ = {rho:.3f}\np = {p_corr:.3f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Coleman Homophily', fontsize=12, fontweight='bold')
    ax.set_ylabel('Burt Constraint', fontsize=12, fontweight='bold')
    ax.set_title('Homophily vs Constraint by Department', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Homophily-constraint scatter plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_sample_scatter_data(num_nodes: int = 100) -> tuple:
    """Create sample data for testing scatter plot"""
    np.random.seed(42)
    
    nodes = [f'node_{i}' for i in range(num_nodes)]
    departments = ['DCAR', 'DG', 'DISQ', 'DMCT', 'DMI', 'DSE', 'DST', 'SCOM', 'SDOC', 'SFLE']
    
    # Generate homophily values (bounded between -1 and 1)
    homophily = np.random.beta(2, 2, num_nodes) * 2 - 1
    
    # Generate constraint values (bounded between 0 and 1)
    constraint = np.random.beta(1, 3, num_nodes)
    
    # Add some correlation
    constraint = constraint + 0.2 * (homophily + 1) / 2
    constraint = np.clip(constraint, 0, 1)
    
    # Create DataFrames
    homophily_df = pd.Series(homophily, index=nodes, name='homophily')
    constraint_df = pd.Series(constraint, index=nodes, name='constraint')
    
    # Random department assignment
    node_depts = {node: np.random.choice(departments) for node in nodes}
    
    return homophily_df, constraint_df, node_depts
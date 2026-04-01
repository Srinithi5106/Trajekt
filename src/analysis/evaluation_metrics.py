"""
evaluation_metrics.py
====================
Comprehensive evaluation metrics for multi-layer network analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import shap
from scipy import stats
from typing import Dict, Tuple, List
import networkx as nx

class NetworkEvaluator:
    def __init__(self, email_graph: nx.Graph, proximity_graph: nx.Graph, node_departments: Dict[str, str]):
        self.email_graph = email_graph
        self.proximity_graph = proximity_graph
        self.node_departments = node_departments
    
    def compute_auc_roc(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute AUC-ROC for link prediction tasks"""
        return roc_auc_score(true_labels, predictions)
    
    def compute_precision_at_k(self, predictions: np.ndarray, true_labels: np.ndarray, k: int = 10) -> float:
        """Compute Precision@K for top-K predictions"""
        top_k_indices = np.argsort(predictions)[-k:]
        return np.mean(true_labels[top_k_indices])
    
    def compute_spearman_correlation(self, metric1: Dict[str, float], metric2: Dict[str, float]) -> Tuple[float, float]:
        """Compute Spearman correlation between node-level metrics"""
        common_nodes = set(metric1.keys()) & set(metric2.keys())
        vals1 = [metric1[node] for node in common_nodes if not np.isnan(metric1[node])]
        vals2 = [metric2[node] for node in common_nodes if not np.isnan(metric2[node])]
        
        if len(vals1) < 3:
            return (np.nan, np.nan)
        
        rho, p_value = stats.spearmanr(vals1, vals2)
        return (rho, p_value)
    
    def compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for Burt constraint inequality"""
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        return gini
    
    def compute_temporal_coleman_h(self, temporal_graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
        """Compute Coleman homophily over time periods"""
        from src.analysis.homophily import coleman_homophily, aggregate_by_dept
        
        results = []
        for period, G in temporal_graphs.items():
            # Add department attributes
            for node in G.nodes():
                G.nodes[node]['dept'] = self.node_departments.get(node, 'unknown')
            
            h_dict = coleman_homophily(G)
            dept_h = aggregate_by_dept(h_dict, G)
            
            for dept, h_val in dept_h.items():
                results.append({'period': period, 'department': dept, 'coleman_h': h_val})
        
        return pd.DataFrame(results)
    
    def train_gbm_with_shap(self, features: pd.DataFrame, target: pd.Series) -> Tuple[GradientBoostingClassifier, np.ndarray]:
        """Train GBM and compute SHAP values for interpretability"""
        gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gbm.fit(features, target)
        
        explainer = shap.TreeExplainer(gbm)
        shap_values = explainer.shap_values(features)
        
        return gbm, shap_values
    
    def ablation_study(self, features: pd.DataFrame, target: pd.Series, 
                      feature_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """Ablation study: static vs temporal features"""
        results = {}
        
        # Full model
        results['full_model'] = np.mean(cross_val_score(
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            features, target, cv=5, scoring='roc_auc'
        ))
        
        # Static features only
        static_features = [col for col in features.columns if col in feature_groups.get('static', [])]
        if static_features:
            results['static_only'] = np.mean(cross_val_score(
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                features[static_features], target, cv=5, scoring='roc_auc'
            ))
        
        # Temporal features only
        temporal_features = [col for col in features.columns if col in feature_groups.get('temporal', [])]
        if temporal_features:
            results['temporal_only'] = np.mean(cross_val_score(
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                features[temporal_features], target, cv=5, scoring='roc_auc'
            ))
        
        return results
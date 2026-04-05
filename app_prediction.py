import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from pathlib import Path

st.set_page_config(page_title="Node Risk Prediction", layout="wide")

MODEL_DIR = Path("data/models")
FEATURES_CSV = Path("data/features.csv")

# ---------------------------------------------------------------------------
# Load Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models_and_data():
    if not FEATURES_CSV.exists() or not (MODEL_DIR / "shap_values.pkl").exists():
        return None, None, None, None

    # Load baseline features data
    df = pd.read_csv(FEATURES_CSV)
    
    # Load SHAP and Meta data
    with open(MODEL_DIR / "shap_values.pkl", "rb") as f:
        meta = pickle.load(f)
        
    shap_vals = meta.get("shap_values")
    X_sample = meta.get("X_sample")
    
    # Load predictions if available
    pred_path = MODEL_DIR / "predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
    else:
        pred_df = df.copy() # fallback
        pred_df["risk_score"] = 0.0
        
    return df, pred_df, shap_vals, X_sample


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
df, pred_df, shap_vals, X_sample = load_models_and_data()

st.title("Node Risk Prediction (GBM)")

if df is None:
    st.error("Model files or features not found. Please run ML model training first.")
    st.stop()

st.sidebar.header("Filter Predictions")
min_risk = st.sidebar.slider("Minimum Risk Score Threshold", 0.0, 1.0, 0.5)

tab1, tab2, tab3 = st.tabs([
    "Node Risk Explorer", 
    "SHAP Explainability", 
    "At-Risk Node Table"
])

# ---------------------------------------------------------------------------
# TAB 1: Node Risk Explorer
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Risk Score Distribution")
    fig1 = px.histogram(pred_df, x="risk_score", color="label", nbins=50,
                        title="Distribution of Predicted Risk Scores (colored by True Label)",
                        labels={"risk_score": "Probability of Risk", "label": "Actual Vol Drop > 70%"})
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Nodes Assessed", len(pred_df))
    with col2:
        high_risk_count = len(pred_df[pred_df["risk_score"] >= min_risk])
        st.metric(rf"Nodes $\ge$ {min_risk} Risk", high_risk_count)


# ---------------------------------------------------------------------------
# TAB 2: SHAP Explainability
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Global Feature Importance (SHAP)")
    if shap_vals is not None and X_sample is not None and len(shap_vals) > 0:
        # shap_values from TreeExplainer might be a list (multiclass) or array (binary/regression)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_names = [c for c in df.columns if c not in ("node", "label")]
        
        # Guard in case scaler changes column counts
        if len(feature_names) == len(mean_abs_shap):
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP| (Impact)": mean_abs_shap
            }).sort_values("Mean |SHAP| (Impact)", ascending=True)
            
            fig2 = px.bar(shap_df, x="Mean |SHAP| (Impact)", y="Feature", orientation='h',
                          title="Average Impact on Model Output Magnitude")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("SHAP shape mismatch with features.")
            
        st.markdown("---")
        st.subheader("Individual Node Explainability (Local SHAP)")
        node_ids = df["node"].iloc[:len(sv)].tolist()
        selected_node = st.selectbox("Select Node ID to Explain", node_ids)
        if selected_node:
            node_idx = node_ids.index(selected_node)
            node_sv = sv[node_idx]
            
            waterfall_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": node_sv
            })
            waterfall_df["Direction"] = waterfall_df["SHAP Value"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
            waterfall_df = waterfall_df.sort_values(by="SHAP Value")
            
            fig3 = px.bar(waterfall_df, x="SHAP Value", y="Feature", orientation='h', 
                          color="Direction", color_discrete_map={"Increases Risk": "#ef553b", "Decreases Risk": "#00cc96"},
                          title=f"Feature Contributions to Risk Score for {selected_node}")
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("SHAP values not available in the model pickle.")


# ---------------------------------------------------------------------------
# TAB 3: At-Risk Table
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(rf"Nodes with Risk Score $\ge$ {min_risk}")
    
    high_risk_df = pred_df[pred_df["risk_score"] >= min_risk].sort_values("risk_score", ascending=False)
    
    # Select columns to show
    cols_to_show = ["node", "risk_score", "label", "tb_drop", "tb_final", "degree", "cross_closure"]
    cols_to_show = [c for c in cols_to_show if c in high_risk_df.columns]
    
    st.dataframe(high_risk_df[cols_to_show], use_container_width=True)
    
    st.download_button(
        label="Download High Risk Nodes as CSV",
        data=high_risk_df.to_csv(index=False).encode('utf-8'),
        file_name='high_risk_nodes.csv',
        mime='text/csv',
    )

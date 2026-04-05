import streamlit as st
import networkx as nx
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import tempfile
import tempfile
import tempfile # Needed for PyVis iframe
import os

from src.data_loader import get_email_graph, get_proximity_graph, get_monthly_snapshots
from src.analysis.triadic_closure import clustering_by_dept, cross_layer_closure_rate
from src.analysis.homophily import coleman_homophily
from src.analysis.structural_holes import burt_constraint
from src.analysis.temporal_betweenness import temporal_betweenness_per_snapshot, build_tb_matrix


st.set_page_config(page_title="Network Analysis", layout="wide")

# ---------------------------------------------------------------------------
# Caching Data/Computation
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_graphs():
    G_e = get_email_graph()
    G_p = get_proximity_graph()
    snaps = get_monthly_snapshots()
    return G_e, G_p, snaps

@st.cache_data(show_spinner=True)
def get_closure_data(_G_e, _G_p):
    df_clust_e = clustering_by_dept(_G_e)
    df_clust_e["Layer"] = "Email"
    df_clust_p = clustering_by_dept(_G_p)
    df_clust_p["Layer"] = "Proximity"
    
    df_clust = pd.concat([df_clust_e, df_clust_p], ignore_index=True)
    df_cross = cross_layer_closure_rate(_G_e, _G_p)
    return df_clust, df_cross

@st.cache_data(show_spinner=True)
def get_homophily_holes(_G_e, _G_p):
    h_e = coleman_homophily(_G_e, dept_attr="dept", weight_attr="weight")
    h_p = coleman_homophily(_G_p, dept_attr="dept", weight_attr="weight")
    
    c_e = burt_constraint(_G_e, weight_attr="weight")
    c_p = burt_constraint(_G_p, weight_attr="weight")
    
    def _build_df(h_dict, c_dict, layer_name):
        records = []
        for node in h_dict.keys():
            if node in c_dict:
                records.append({
                    "Node": str(node),
                    "Homophily": h_dict[node],
                    "Constraint": c_dict[node],
                    "Layer": layer_name
                })
        return pd.DataFrame(records)
    
    df_e = _build_df(h_e, c_e, "Email")
    df_p = _build_df(h_p, c_p, "Proximity")
    return pd.concat([df_e, df_p], ignore_index=True)

@st.cache_data(show_spinner=True)
def get_tb_data(_snaps):
    tb_df = temporal_betweenness_per_snapshot(_snaps)
    tb_mat = build_tb_matrix(tb_df)
    return tb_df, tb_mat


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
G_e, G_p, snaps = load_graphs()

st.title("Multi-Layer Network Analysis")

st.sidebar.header("Global Filters")
layer_choice = st.sidebar.selectbox("Select Layer for Visualization", ["Email", "Proximity"])
G_active = G_e if layer_choice == "Email" else G_p

# Tabs config
tab1, tab2, tab3, tab4 = st.tabs([
    "Network Overview", 
    "Triadic Closure", 
    "Homophily & Structural Holes", 
    "Temporal Betweenness"
])

# ---------------------------------------------------------------------------
# TAB 1: Network Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader(f"{layer_choice} Network Topology")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Nodes", G_active.number_of_nodes())
        st.metric("Edges", G_active.number_of_edges())
        st.metric("Departments", len(set(nx.get_node_attributes(G_active, "dept").values())))
    
    with col2:
        max_nodes = 150
        if G_active.number_of_nodes() > max_nodes:
            st.warning(f"Showing top 150 nodes by activity. Full graph has {G_active.number_of_nodes()} nodes.")
            degrees = dict(G_active.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G_vis = G_active.subgraph(top_nodes)
        else:
            G_vis = G_active

        @st.cache_data(show_spinner=False)
        def get_pyvis_html(nodes, edges, layer):
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
            net.set_options("""
            var options = {
              "physics": {
                "barnesHut": { "gravitationalConstant": -8000, "springLength": 120 },
                "stabilization": { "iterations": 100, "onlyDynamicEdges": false }
              }
            }
            """)
            for node, data in nodes:
                net.add_node(str(node), title=f"Dept: {data.get('dept', 'Unknown')}")
            for u, v, data in edges:
                net.add_edge(str(u), str(v), value=data.get('weight', 1))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html = f.read()
            os.unlink(tmp.name)
            return html

        html_data = get_pyvis_html(list(G_vis.nodes(data=True)), list(G_vis.edges(data=True)), layer_choice)
        st.components.v1.html(html_data, height=600, scrolling=False)


# ---------------------------------------------------------------------------
# TAB 2: Triadic Closure
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Triadic Closure by Department")
    df_clust, df_cross = get_closure_data(G_e, G_p)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_clust, x="dept", y="mean_clustering", color="Layer", barmode="group",
                      title="Mean Clustering Coefficient by Department")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if not df_cross.empty:
            fig2 = px.bar(df_cross, x="dept", y="closure_rate", 
                          title="Cross-Layer Closure Rate by Department",
                          labels={"closure_rate": "Closure Rate (Email $\\cap$ Proximity)"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No cross-layer triads found.")
            

# ---------------------------------------------------------------------------
# TAB 3: Homophily & Structural Holes
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Homophily vs. Burt's Constraint")
    df_hh = get_homophily_holes(G_e, G_p)
    
    if not df_hh.empty:
        fig3 = px.scatter(df_hh, x="Homophily", y="Constraint", color="Layer", hover_data=["Node"],
                          title="Homophily (Coleman) vs. Structural Holes (Burt's Constraint)",
                          opacity=0.7)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Pearson correlation
        e_corr = df_hh[df_hh["Layer"] == "Email"][["Homophily", "Constraint"]].corr()
        p_corr = df_hh[df_hh["Layer"] == "Proximity"][["Homophily", "Constraint"]].corr()
        
        st.write("**Correlations (Homophily ~ Constraint):**")
        st.write(f"- Email Layer: {e_corr.iloc[0, 1]:.3f}" if len(df_hh[df_hh["Layer"]=="Email"]) > 1 else "- Email Layer: N/A")
        st.write(f"- Proximity Layer: {p_corr.iloc[0, 1]:.3f}" if len(df_hh[df_hh["Layer"]=="Proximity"]) > 1 else "- Proximity Layer: N/A")
    else:
        st.info("Not enough data to scatter.")


# ---------------------------------------------------------------------------
# TAB 4: Temporal Betweenness
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Temporal Betweenness Centrality")
    if not snaps:
        st.warning("No monthly snapshots available.")
    else:
        tb_df, tb_mat = get_tb_data(snaps)
        
        if not tb_mat.empty:
            # Heatmap of top 30 nodes by overall mean TB
            top_nodes = tb_mat.mean(axis=1).nlargest(30).index
            tb_top = tb_mat.loc[top_nodes]
            
            fig4 = px.imshow(tb_top, labels=dict(x="Month", y="Node", color="Betweenness"),
                             title="Betweenness Centrality over Time (Top 30 Nodes)",
                             aspect="auto", color_continuous_scale="Viridis")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No betweenness data available.")

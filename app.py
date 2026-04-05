import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
import io

# --- CRITICAL FIX: Image Decompression Bomb ---
Image.MAX_IMAGE_PIXELS = None 

# Check for required visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 1. Page Configuration
st.set_page_config(
    page_title="Trajekt Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Styling
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg-main: #ffffff;
        --bg-card: #ffffff;
        --border-card: #e2e8f0;
        --text-primary: #000000;
        --accent-blue: #0052ff;
    }

    /* REMOVE TOP BLACK BAR / STREAMLIT HEADER */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0% !important;
    }

    .stApp { background-color: var(--bg-main); color: #000000 !important; }

    /* HIGH CONTRAST TEXT OVERRIDE FOR MAIN CONTENT */
    .main p, .main div, .main span, .main label, .main small {
        color: #000000 !important;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #000000 !important;
        margin-top: 0rem !important;
        margin-bottom: 0.1rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling - Dark Theme with White Text */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important; 
        border-right: 1px solid #1e293b;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    /* FIX TABS CONTRAST */
    button[data-baseweb="tab"] p {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    button[aria-selected="true"] p {
        color: var(--accent-blue) !important;
    }

    /* COMPACT CARDS */
    .glass-card {
        background: var(--bg-card);
        border: 2px solid #f1f5f9;
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        overflow: hidden; 
        min-height: fit-content;
    }

    /* METRIC CARDS - Responsive for longer text */
    .metric-card {
        min-height: 85px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    
    .metric-card h2 {
        color: var(--accent-blue) !important;
        font-size: 1.25rem !important;
        margin: 2px 0 !important;
    }

    .metric-card small {
        text-transform: uppercase;
        letter-spacing: 0.02em;
        font-size: 0.55rem;
        font-weight: 800;
        line-height: 1.2;
        display: block;
    }

    .metric-sublabel {
        font-size: 0.5rem !important;
        color: #64748b !important;
        font-style: italic;
        margin-top: -2px;
    }

    .chart-header {
        display: flex;
        align-items: center;
        padding-bottom: 1px;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 4px;
    }

    .js-plotly-plot .plotly text {
        fill: #000000 !important;
        font-weight: 600 !important;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Data Processing
@st.cache_data
def load_analytics_data():
    depts = ["Engineering", "Product", "Sales", "HR", "Design", "Marketing"]
    data = pd.DataFrame({
        "Department": np.random.choice(depts, 2000),
        "Homophily": np.random.uniform(-0.75, 1.25, 2000),
        "Constraint": np.random.uniform(0.05, 0.45, 2000),
        "Closure_Rate": np.random.uniform(0, 1, 2000),
        "Engagement": np.random.randint(10, 100, 2000)
    })
    return data

# 4. Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <i class="fa-solid fa-bolt-lightning fa-2xl" style="color: #0052ff;"></i>
            <h2 style="letter-spacing: 2px; margin-top: 10px; font-size: 1.2rem; color: #ffffff;">TRAJEKT</h2>
        </div>
    """, unsafe_allow_html=True)

    selected_depts = st.multiselect(
        "Focus Departments", 
        ["Engineering", "Product", "Sales", "HR", "Design", "Marketing"],
        default=["Engineering", "Product", "Sales"]
    )
    
    st.divider()
    st.markdown("### System Controls")
    st.toggle("High Performance Mode", value=True)

# 5. Header
st.markdown('<h1 style="font-size: 1.7rem;">Analytics Engine <span style="color:#0052ff;">v4.2</span></h1>', unsafe_allow_html=True)

# 6. Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

def custom_metric(col, label, sublabel, value, icon):
    with col:
        st.markdown(f"""
            <div class="glass-card metric-card">
                <i class="fa-solid {icon}" style="color:#0052ff; font-size: 0.8rem; margin-bottom: 4px;"></i>
                <small>{label}</small>
                <div class="metric-sublabel">{sublabel}</div>
                <h2 style="line-height: 1.1;">{value}</h2>
            </div>
        """, unsafe_allow_html=True)

# Updated metrics with scientific names
custom_metric(col_m1, "Email Homophily", "(Coleman index rho)", "0.68", "fa-envelope")
custom_metric(col_m2, "Proximity Homophily", "(Highly Sig Correlation)", "0.31", "fa-street-view")
custom_metric(col_m3, "Email Constraint", "(Burts Structural holes)", "0.45", "fa-lock")
custom_metric(col_m4, "Proximity Constraint", "(Burts Structural holes)", "0.22", "fa-expand-arrows-alt")

# 7. Main Dashboard Grid
col_left, col_right = st.columns(2)

def smart_render_image(title, icon, path, container, force_live=False):
    with container:
        st.markdown(f'''
            <div class="glass-card">
                <div class="chart-header">
                    <i class="fa-solid {icon}" style="color:#0052ff; margin-right:8px; font-size:0.75rem;"></i>
                    <h3 style="font-size:0.8rem; color:#000000;">{title}</h3>
                </div>
        ''', unsafe_allow_html=True)

        asset_rendered = False
        # Try to render from file first
        if not force_live and os.path.exists(path):
            try:
                with Image.open(path) as img:
                    img.thumbnail((800, 800))
                    st.image(img, use_container_width=True)
                    asset_rendered = True
            except:
                pass
        
        # Fallback to Live Plotly logic if file missing or force_live is True
        if not asset_rendered:
            if PLOTLY_AVAILABLE:
                df = load_analytics_data()
                if "Heatmap" in title or "Temporal" in title:
                    fig = px.density_heatmap(df, x="Homophily", y="Constraint", 
                                           height=160, color_continuous_scale='Blues')
                    fig.update_layout(coloraxis_showscale=False)
                else:
                    fig = px.scatter(df, x="Homophily", y="Constraint", height=160, 
                                   color_discrete_sequence=['#0052ff'])
                
                fig.update_layout(
                    margin=dict(l=5, r=5, t=5, b=5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="black", size=9)
                )
                fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=False)
                fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Plotly not installed. Could not render live chart.")

        st.markdown('</div>', unsafe_allow_html=True)

smart_render_image("Network Homophily Matrix", "fa-diagram-project", "outputs/stage4_homophily_constraint_scatter.png", col_left)
smart_render_image("Multilayer Topology Map", "fa-network-wired", "outputs/stage4_multilayer_graph.png", col_right)
smart_render_image("Closure Velocity Analysis", "fa-infinity", "outputs/stage4_cross_layer_closure.png", col_left)
smart_render_image("Temporal Activity Heatmap", "fa-fire-flame-curved", "outputs/stage4_temporal_heatmap.png", col_right)

# 8. Distribution Intelligence Section
st.markdown('<div class="glass-card"><h3 style="font-size:0.85rem; margin-bottom:4px;">Distribution Intelligence</h3>', unsafe_allow_html=True)

df_all = load_analytics_data()
df_filtered = df_all[df_all["Department"].isin(selected_depts)]

tab1, tab2 = st.tabs(["Interactive Analytics", "Raw Matrix Explorer"])

with tab1:
    if PLOTLY_AVAILABLE:
        # Grouping to ensure the bar chart has data
        chart_data = df_filtered.groupby("Department")["Engagement"].mean().reset_index()
        if not chart_data.empty:
            fig = px.bar(chart_data, 
                         x="Department", y="Engagement", height=220,
                         color_discrete_sequence=['#0052ff'])
            fig.update_layout(
                margin=dict(l=5, r=5, t=20, b=5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="black", size=10)
            )
            fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select departments in the sidebar to see data.")
    else:
        st.error("Visualization Error: 'plotly' library is missing. Please run 'pip install plotly'.")

with tab2:
    st.dataframe(df_filtered, height=220, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <p style="text-align:center; color:#000000; padding:10px 0; font-size: 0.7rem; font-weight: 600;">
        <i class="fa-solid fa-code"></i> 
        TRAJEKT ENGINE v4.2 • High Contrast Mode • Optimized Node Grid
    </p>
""", unsafe_allow_html=True)
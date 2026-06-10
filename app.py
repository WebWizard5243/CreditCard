"""
app.py — Modern UI for Fraud Shield AI
"""

from __future__ import annotations

import time
import logging
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import DataPipeline
from graph_engine import GraphEngine
from model import FraudClassifier

st.set_page_config(page_title="Fraud Shield AI", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e293b; }
[data-testid="stSidebar"] { background-color: #0b1120 !important; border-right: none; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; }
div[data-testid="stMetric"] label { color: #64748b !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.75rem !important; letter-spacing: 0.5px; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; font-size: 1.8rem !important; }
/* Custom sidebar logo */
.sidebar-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 2rem; padding: 1rem 0; }
.sidebar-logo h2 { margin: 0; font-size: 1.2rem; font-weight: 800; letter-spacing: 1px; color: white; }
.sidebar-logo span { font-size: 0.7rem; color: #94a3b8; font-weight: 400; text-transform: uppercase; letter-spacing: 1.5px; }
/* Override radio buttons in sidebar */
div[role="radiogroup"] > label { padding: 10px 15px; border-radius: 8px; margin-bottom: 5px; cursor: pointer; transition: background 0.2s; }
div[role="radiogroup"] > label:hover { background-color: rgba(255,255,255,0.05); }
.st-c7 {background-color: transparent !important;}
/* Fix checkbox tick visibility */
/* Checked Box Container background & border */
[data-baseweb="checkbox"] input[type="checkbox"]:checked + *,
[data-checked="true"] input + *,
[data-checked="true"] [role="checkbox"],
[aria-checked="true"] + *,
[data-checked="true"] > span > span:first-child {
    background-color: #ffffff !important;
    border-color: #3b82f6 !important;
}

/* Color the checkmark (tick) blue globally inside any checkbox */
[data-baseweb="checkbox"] svg,
[data-baseweb="checkbox"] svg *,
[data-testid="stCheckbox"] svg,
[data-testid="stCheckbox"] svg *,
[data-checked="true"] svg,
[data-checked="true"] svg *,
[aria-checked="true"] ~ svg,
[aria-checked="true"] ~ * svg,
[aria-checked="true"] ~ * svg * {
    color: #3b82f6 !important;
    stroke: #3b82f6 !important;
}
[data-baseweb="checkbox"] svg polyline,
[data-testid="stCheckbox"] svg polyline,
[data-checked="true"] svg polyline,
[aria-checked="true"] ~ * svg polyline {
    fill: none !important;
    stroke: #3b82f6 !important;
}
[data-baseweb="checkbox"] svg path,
[data-testid="stCheckbox"] svg path,
[data-checked="true"] svg path,
[aria-checked="true"] ~ * svg path {
    fill: #3b82f6 !important;
    stroke: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown('''
<div class="sidebar-logo">
    <div>
        <h2>🛡️ FRAUD SHIELD AI</h2>
        <span>Graph + ML</span>
    </div>
</div>
''', unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Dashboard", "Prediction", "Analytics", "Graph Network", "Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
if "clf" in st.session_state:
    st.sidebar.markdown("**Model Status:** 🟢 Active")
    st.sidebar.markdown(f"**Model Accuracy:** {st.session_state['results']['accuracy']*100:.1f}%")
else:
    st.sidebar.markdown("**Model Status:** 🔴 Not Trained")
    st.sidebar.markdown("**Model Accuracy:** --")

# --- Helper Functions ---
def check_trained():
    required_keys = ["clf", "ge", "pipeline", "raw_df", "results", "template_row"]
    if any(k not in st.session_state for k in required_keys):
        st.warning("Model is not trained yet or the session was reset. Please go to **Settings** and click **Run Full Pipeline**.")
        st.stop()

# =====================================================================
# DASHBOARD
# =====================================================================
if page == "Dashboard":
    st.title("Fraud Detection Dashboard")
    st.markdown("Monitor transactions and detect fraud using Graph Analysis and Machine Learning.")
    
    check_trained()
    
    raw_df = st.session_state["raw_df"]
    results = st.session_state["results"]
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_tx = len(raw_df)
    fraud_tx = raw_df["is_fraud"].sum()
    fraud_rate = (fraud_tx / total_tx) * 100
    
    with m1:
        st.container(border=True).metric("Total Transactions", f"{total_tx:,}")
    with m2:
        st.container(border=True).metric("Fraud Transactions", f"{fraud_tx:,}")
    with m3:
        st.container(border=True).metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with m4:
        st.container(border=True).metric("Model Accuracy", f"{results['accuracy']*100:.1f}%")
        
    c1, c2 = st.columns([2, 1])
    
    with c1:
        with st.container(border=True):
            st.markdown("**Transactions vs Fraud Cases Over Time**")
            # Create a time series by converting unix_time to datetime
            df_ts = raw_df.copy()
            df_ts['date'] = pd.to_datetime(df_ts['unix_time'], unit='s').dt.date
            daily_stats = df_ts.groupby('date').agg(Total=('is_fraud', 'count'), Fraud=('is_fraud', 'sum')).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['Total'], mode='lines+markers', name='Total Transactions', line=dict(color='#3b82f6')))
            fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['Fraud'], mode='lines+markers', name='Fraud Transactions', line=dict(color='#ef4444')))
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            st.markdown("**Transaction Distribution**")
            fig = px.pie(names=['Legitimate', 'Fraud'], values=[total_tx - fraud_tx, fraud_tx], hole=0.6, color_discrete_sequence=['#3b82f6', '#ef4444'])
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300, showlegend=True)
            # Add center text
            fig.add_annotation(text=f"<b>{fraud_rate:.2f}%</b><br>Fraud", x=0.5, y=0.5, font=dict(size=20), showarrow=False)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PREDICTION
# =====================================================================
elif page == "Prediction":
    st.title("Graph-Based Fraud Detection")
    st.markdown("Enter transaction details to analyze fraud probability using Graph Analysis and Machine Learning.")
    
    check_trained()
    
    raw_df = st.session_state["raw_df"]
    real_merchants = sorted(raw_df["merchant"].unique().tolist())
    
    # Build city lookup table for merchant city selection
    city_lookup = raw_df.groupby("city").agg({
        "lat": "mean", "long": "mean", "city_pop": "first", "state": "first"
    }).reset_index().sort_values("city")
    city_names = city_lookup["city"].tolist()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("**Transaction Details**")
            st.caption("Enter a card number and transaction details. The model analyzes amount, location, category, and behavioral patterns.")
            with st.form("prediction_form"):
                test_cc_num = st.text_input("Card Number", value="4024007100468820")
                test_merchant = st.text_input("Merchant Name", value="fraud_Rippin Inc")
                test_category = st.selectbox("Category", options=sorted(raw_df["category"].unique().tolist()))
                test_amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=10.0)
                test_merch_city = st.text_input("Merchant City", value="Dallas")
                
                submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)

    with col2:
        if submitted:
            pipeline = st.session_state["pipeline"]
            ge = st.session_state["ge"]
            clf = st.session_state["clf"]
            
            final_merchant = test_merchant
            
            # Look up merchant city coordinates
            merch_city_rows = city_lookup[city_lookup["city"].str.lower() == test_merch_city.strip().lower()]
            is_known_city = True
            if len(merch_city_rows) > 0:
                merch_city_data = merch_city_rows.iloc[0]
                test_merch_lat = float(merch_city_data["lat"])
                test_merch_long = float(merch_city_data["long"])
            else:
                test_merch_lat = float(raw_df["lat"].median())
                test_merch_long = float(raw_df["long"].median())
                is_known_city = False
            
            # Try to find a real row for this card to use as a template
            card_rows = raw_df[raw_df["cc_num"].astype(str) == str(test_cc_num)]
            if len(card_rows) > 0:
                template = card_rows.iloc[0].to_dict()
                is_known_card = True
            else:
                template = st.session_state["template_row"].copy()
                is_known_card = False

            # Override with user inputs
            template["amt"] = test_amt
            template["category"] = test_category
            template["cc_num"] = test_cc_num
            template["merchant"] = final_merchant
            # Use a unix_time from within the training data range (not current time!)
            template["unix_time"] = int(raw_df["unix_time"].median())
            template["merch_lat"] = test_merch_lat
            template["merch_long"] = test_merch_long
            if "is_fraud" in template:
                del template["is_fraud"]

            test_df = pd.DataFrame([template])

            try:
                processed_test = pipeline.preprocess(test_df, fit=False)
                enriched_test = ge.extract_features(processed_test)
                
                # For unknown cards/merchants, impute graph features with dataset medians
                if not is_known_card:
                    enriched_df = st.session_state.get("enriched_df_stats")
                    if enriched_df is not None:
                        for col in ["card_degree", "merchant_degree", "card_tx_frequency", "card_unique_merchants"]:
                            if col in enriched_df:
                                enriched_test[col] = enriched_df[col]
                    else:
                        # Fallback: use reasonable averages
                        enriched_test["card_degree"] = 600
                        enriched_test["merchant_degree"] = 800
                        enriched_test["card_tx_frequency"] = 2
                        enriched_test["card_unique_merchants"] = 50
                
                X_test, _ = clf.prepare_features(enriched_test)
                proba = clf.predict_proba(X_test)[0] * 100
                
                with st.container(border=True):
                    st.markdown("**Analysis Result**")
                    if not is_known_card:
                        st.caption("This card is not in the training data. Graph features were estimated using dataset averages.")
                    if not is_known_city:
                        st.caption(f"The city '{test_merch_city}' is not in the dataset. Location coordinates were estimated using dataset averages.")
                    rcol1, rcol2 = st.columns([1, 1])
                    with rcol1:
                        if proba >= 50:
                            st.markdown("<h3 style='color:#ef4444;'>Fraud Detected</h3>", unsafe_allow_html=True)
                            st.markdown("This transaction is likely to be fraudulent.")
                            st.markdown("<span style='background:#fef2f2; color:#ef4444; padding:5px 10px; border-radius:5px; font-weight:bold;'>HIGH RISK</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h3 style='color:#22c55e;'>Legitimate</h3>", unsafe_allow_html=True)
                            st.markdown("This transaction appears safe.")
                            st.markdown("<span style='background:#f0fdf4; color:#22c55e; padding:5px 10px; border-radius:5px; font-weight:bold;'>LOW RISK</span>", unsafe_allow_html=True)
                    
                    with rcol2:
                        color = '#ef4444' if proba >= 50 else '#22c55e'
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = proba,
                            number = {'suffix': "%", 'font': {'size': 40, 'color': '#0f172a'}},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [None, 100], 'visible': False},
                                'bar': {'color': color},
                                'bgcolor': "rgba(0,0,0,0.05)",
                                'borderwidth': 0,
                            }
                        ))
                        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=150)
                        st.plotly_chart(fig, use_container_width=True)
                
                with st.container(border=True):
                    st.markdown("**Feature Breakdown**")
                    gf1, gf2 = st.columns(2)
                    gf1.metric("Card Degree", int(enriched_test.iloc[0]['card_degree']))
                    gf2.metric("Merchant Degree", int(enriched_test.iloc[0]['merchant_degree']))
                    gf1.metric("Tx Frequency (1hr)", int(enriched_test.iloc[0]['card_tx_frequency']))
                    gf2.metric("Unique Merchants", int(enriched_test.iloc[0]['card_unique_merchants']))
                    
                    # Show distance between cardholder and merchant
                    card_lat = template.get("lat", 0)
                    card_long = template.get("long", 0)
                    dist_km = ((card_lat - test_merch_lat)**2 + (card_long - test_merch_long)**2)**0.5 * 111
                    gf1.metric("Distance (est.)", f"{dist_km:.0f} km")
                    gf2.metric("Card Status", "Known" if is_known_card else "Unknown")
                    
                    if proba >= 50:
                        st.markdown("<div style='background:#fef2f2; color:#ef4444; padding:10px; border-radius:5px; border: 1px solid #fecaca; text-align:center;'>High risk factors detected in this transaction.</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            with st.container(border=True):
                st.info("Submit the form to see the analysis result.")

# =====================================================================
# ANALYTICS
# =====================================================================
elif page == "Analytics":
    st.title("Analytics Dashboard")
    st.markdown("Insights and trends from transaction data.")
    
    check_trained()
    
    raw_df = st.session_state["raw_df"]
    results = st.session_state["results"]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.container(border=True).metric("Legitimate Transactions", f"{len(raw_df) - raw_df['is_fraud'].sum():,}")
    with m2:
        st.container(border=True).metric("Fraud Transactions", f"{raw_df['is_fraud'].sum():,}")
    with m3:
        st.container(border=True).metric("Recall", f"{results['recall']*100:.1f}%")
    with m4:
        st.container(border=True).metric("Precision", f"{results['precision']*100:.1f}%")
        
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Fraud vs Legitimate**")
            fig = px.pie(names=['Legitimate', 'Fraud'], values=[len(raw_df)-raw_df['is_fraud'].sum(), raw_df['is_fraud'].sum()], hole=0.6, color_discrete_sequence=['#3b82f6', '#ef4444'])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, showlegend=True, legend=dict(yanchor="bottom", y=0.01, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            st.markdown("**Top 5 Fraudulent Merchants**")
            top_merchants = raw_df[raw_df['is_fraud'] == 1]['merchant'].value_counts().head(5).reset_index()
            top_merchants.columns = ['Merchant', 'Count']
            fig = px.bar(top_merchants, x='Count', y='Merchant', orientation='h', color_discrete_sequence=['#3b82f6'])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
    with c3:
        with st.container(border=True):
            st.markdown("**Model Performance**")
            perf_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                "Value": [f"{results['accuracy']:.4f}", f"{results['precision']:.4f}", f"{results['recall']:.4f}", f"{results['f1']:.4f}", f"{results['roc_auc']:.4f}"]
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

# =====================================================================
# GRAPH NETWORK
# =====================================================================
elif page == "Graph Network":
    st.title("Transaction Network Graph")
    st.markdown("Visualizing relationships between Cards, Merchants and Locations.")
    
    check_trained()
    
    st.info("Large scale network graphs (600k+ edges) cannot be perfectly rendered in the browser at once. Showing a sampled sub-graph visualization.")
    
    with st.container(border=True):
        st.markdown("**Graph Statistics**")
        G_stats = st.session_state["ge"].graph
        g1, g2, g3 = st.columns(3)
        g1.metric("Total Nodes", f"{G_stats.number_of_nodes():,}")
        g2.metric("Total Edges", f"{G_stats.number_of_edges():,}")
        g3.metric("Node Types", "Cards / Merchants / Locations")
        
    with st.container(border=True):
        st.markdown("**Interactive Transaction Subgraph**")
        
        G_full = st.session_state["ge"].graph
        
        import networkx as nx
        import random
        
        # --- Build a BALANCED subgraph: 10 cards, ~5 merchants each, with locations ---
        random.seed(42)
        
        card_nodes = [n for n, d in G_full.nodes(data=True) if d.get("type") == "cardholder"]
        seed_cards = random.sample(card_nodes, min(10, len(card_nodes)))
        
        sub_nodes = set()
        sub_edges = []
        
        for card in seed_cards:
            sub_nodes.add(card)
            # Get neighbors (merchants and locations this card connects to)
            neighbors = list(G_full.successors(card))
            merchants = [n for n in neighbors if G_full.nodes[n].get("type") == "merchant"]
            locations = [n for n in neighbors if G_full.nodes[n].get("type") == "location"]
            
            # Pick up to 5 merchants and 3 locations per card
            picked_merchants = random.sample(merchants, min(5, len(merchants)))
            picked_locations = random.sample(locations, min(3, len(locations)))
            
            for m in picked_merchants:
                sub_nodes.add(m)
                sub_edges.append((card, m))
                # Also get the location this merchant connects to
                m_locations = [n for n in G_full.successors(m) if G_full.nodes[n].get("type") == "location"]
                if m_locations:
                    loc = m_locations[0]
                    sub_nodes.add(loc)
                    sub_edges.append((m, loc))
            
            for loc in picked_locations:
                sub_nodes.add(loc)
                sub_edges.append((card, loc))
        
        G_sub = nx.DiGraph()
        for n in sub_nodes:
            G_sub.add_node(n, **G_full.nodes[n])
        G_sub.add_edges_from(sub_edges)
        
        # --- Compute layout ---
        pos = nx.spring_layout(G_sub, k=1.5, iterations=100, seed=42)
        
        # --- Build Plotly traces ---
        edge_x, edge_y = [], []
        for u, v in G_sub.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color='#cbd5e1'),
            hoverinfo='none', showlegend=False
        )
        
        # Separate node traces by type for proper legend
        node_types = {
            "cardholder": {"color": "#3b82f6", "size": 18, "symbol": "circle", "name": "Card (924 total)"},
            "merchant":   {"color": "#ef4444", "size": 14, "symbol": "diamond", "name": "Merchant (693 total)"},
            "location":   {"color": "#10b981", "size": 10, "symbol": "square", "name": "Location (912 total)"},
        }
        
        node_traces = []
        for ntype, style in node_types.items():
            nx_list = [n for n in G_sub.nodes() if G_sub.nodes[n].get("type") == ntype]
            if not nx_list:
                continue
            xs = [pos[n][0] for n in nx_list]
            ys = [pos[n][1] for n in nx_list]
            labels = [str(n).split(":", 1)[-1][:15] for n in nx_list]
            
            node_traces.append(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(size=style["size"], color=style["color"], symbol=style["symbol"], line=dict(width=1.5, color='white')),
                text=labels, hoverinfo='text',
                name=style["name"]
            ))
        
        fig = go.Figure(data=[edge_trace] + node_traces)
        fig.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# SETTINGS
# =====================================================================
elif page == "Settings":
    st.title("Settings & Configuration")
    st.markdown("Configure model hyperparameters and train the Fraud Shield AI pipeline.")
    
    with st.container(border=True):
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload the credit card transactions CSV",
            type=["csv"],
            help="Expected columns: cc_num, merchant, category, amt, unix_time, is_fraud …",
        )
        use_default = st.checkbox("Use default dataset (dataset.csv in project folder)", value=False)
        
    with st.container(border=True):
        st.markdown("### Model Configuration")
        c1, c2 = st.columns(2)
        with c1:
            n_estimators = st.slider("Number of Trees", 50, 500, 200, step=50)
            test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
        with c2:
            max_depth = st.slider("Max Tree Depth", 5, 50, 20, step=5)
            cv_folds = st.slider("CV Folds", 3, 10, 5)

        run_pipeline = st.button("Run Full Pipeline", type="primary", use_container_width=True)

    if run_pipeline:
        with st.spinner("Loading data …"):
            pipeline = DataPipeline()
            if uploaded_file is not None:
                raw_df = pipeline.load_data(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            elif use_default:
                raw_df = pipeline.load_data("dataset.csv")
            else:
                st.error("Please upload a CSV or enable the default dataset checkbox.")
                st.stop()

        st.success(f"Loaded **{raw_df.shape[0]:,}** transactions  ×  **{raw_df.shape[1]}** columns")

        progress = st.progress(0, text="Preprocessing …")
        graph_df = raw_df.copy()
        processed_df = pipeline.preprocess(raw_df, fit=True)
        
        progress.progress(30, text="Building transaction graph …")
        ge = GraphEngine()
        ge.build_graph(graph_df)
        
        progress.progress(50, text="Extracting graph features …")
        enriched_df = ge.extract_features(processed_df)
        
        progress.progress(70, text="Preparing features …")
        clf = FraudClassifier(n_estimators=n_estimators, max_depth=max_depth)
        X, y = clf.prepare_features(enriched_df)
        
        progress.progress(75, text="Training Random Forest …")
        results = clf.train_and_evaluate(X, y, test_size=test_size, cv_folds=cv_folds)
        progress.progress(100, text="Pipeline complete!")

        # Save to session state
        st.session_state["raw_df"] = graph_df
        st.session_state["pipeline"] = pipeline
        st.session_state["ge"] = ge
        st.session_state["clf"] = clf
        st.session_state["results"] = results
        st.session_state["template_row"] = graph_df.iloc[0].to_dict()
        
        # Save median graph features for imputing unknown cards during prediction
        graph_cols = ["card_degree", "merchant_degree", "card_tx_frequency", "card_unique_merchants"]
        st.session_state["enriched_df_stats"] = {col: enriched_df[col].median() for col in graph_cols if col in enriched_df.columns}

        st.balloons()
        st.success("Training Complete! You can now navigate to the **Dashboard** or **Prediction** tabs.")

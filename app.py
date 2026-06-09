"""
app.py — Streamlit Dashboard for Fraudulent Credit Card Detection
==================================================================
Features:
  • CSV upload or default dataset path
  • Full pipeline: preprocessing → graph construction → feature extraction → training
  • Interactive evaluation dashboard: metrics cards, confusion matrix heatmap,
    ROC & PR curves, feature importance bar chart
  • Modern dark-themed UI with custom CSS
"""

from __future__ import annotations

import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# --- Project modules -------------------------------------------------------
from data_pipeline import DataPipeline
from graph_engine import GraphEngine
from model import FraudClassifier

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Metric Cards ---------- */
div[data-testid="stMetric"] {
    /* Rely on Streamlit's native theme for background/text color */
    border: 1px solid rgba(150,150,150,0.2);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.25);
}
div[data-testid="stMetric"] label {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 0.72rem;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-weight: 700;
    font-size: 1.65rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150,150,150,0.1);
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(99,102,241,0.5);
}

/* ---------- Progress bar ---------- */
div.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 999px;
}

/* ---------- Expander headers ---------- */
details summary {
    font-weight: 600;
}

/* ---------- Section headers ---------- */
h1, h2, h3 { 
    /* Let Streamlit handle header colors for light/dark mode */
}

/* ---------- File uploader ---------- */
section[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.3s;
}
section[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.8);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Matplotlib / Seaborn defaults for dark backgrounds
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## :material/security: Fraud Detection")
    st.markdown("---")
    st.markdown("### :material/settings: Model Configuration")
    n_estimators = st.slider("Number of Trees", 50, 500, 200, step=50)
    max_depth = st.slider("Max Tree Depth", 5, 50, 20, step=5)
    test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
    cv_folds = st.slider("CV Folds", 3, 10, 5)
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;opacity:0.7;'>"
        "Built with Streamlit • NetworkX • Scikit-learn</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800;
                   background: linear-gradient(135deg, #818cf8, #c084fc);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin-bottom:0.2rem;'>
            <span class="material-symbols-rounded" style="vertical-align: middle;">security</span> Credit Card Fraud Detection
        </h1>
        <p style='font-size:1.05rem; margin-top:0; opacity:0.8;'>
            Graph-Augmented Machine Learning Pipeline
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
st.markdown("### :material/folder_open: Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload the credit card transactions CSV",
    type=["csv"],
    help="Expected columns: cc_num, merchant, category, amt, unix_time, is_fraud …",
)

use_default = st.checkbox("Use default dataset (dataset.csv in project folder)", value=True)

run_pipeline = st.button("Run Full Pipeline", icon=":material/rocket_launch:", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_pipeline:
    # ----- Load data -------------------------------------------------------
    with st.spinner("Loading data …"):
        pipeline = DataPipeline()
        if uploaded_file is not None:
            import io
            raw_df = pipeline.load_data(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        elif use_default:
            raw_df = pipeline.load_data("dataset.csv")
        else:
            st.error("Please upload a CSV or enable the default dataset checkbox.")
            st.stop()

    st.success(f":material/check_circle: Loaded **{raw_df.shape[0]:,}** transactions  ×  **{raw_df.shape[1]}** columns")

    # Show class distribution
    fraud_pct = raw_df["is_fraud"].mean() * 100
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Transactions", f"{len(raw_df):,}")
    col_b.metric("Fraudulent", f"{raw_df['is_fraud'].sum():,}")
    col_c.metric("Fraud Rate", f"{fraud_pct:.2f}%")

    # ----- Preprocess ------------------------------------------------------
    progress = st.progress(0, text="Preprocessing …")

    # Keep a copy with graph-relevant columns BEFORE encoding
    graph_df = raw_df.copy()

    processed_df = pipeline.preprocess(raw_df, fit=True)
    progress.progress(25, text="Preprocessing complete ✓")

    # ----- Graph construction & features -----------------------------------
    progress.progress(30, text="Building transaction graph …")
    ge = GraphEngine()
    # Build graph on the original (unencoded) data for meaningful node IDs
    ge.build_graph(graph_df)

    progress.progress(50, text="Extracting graph features …")
    # Merge graph features into the processed dataframe
    # We need cc_num, merchant, unix_time from the processed df
    # but cc_num and merchant are still present (not yet dropped)
    enriched_df = ge.extract_features(processed_df)
    progress.progress(65, text="Graph features extracted ✓")

    # ----- Model training ---------------------------------------------------
    progress.progress(70, text="Preparing features …")
    clf = FraudClassifier(n_estimators=n_estimators, max_depth=max_depth)
    X, y = clf.prepare_features(enriched_df)

    progress.progress(75, text="Training Random Forest …")
    t0 = time.time()
    results = clf.train_and_evaluate(X, y, test_size=test_size, cv_folds=cv_folds)
    train_time = time.time() - t0
    progress.progress(100, text=":material/check_circle: Pipeline complete!")

    st.session_state["pipeline"] = pipeline
    st.session_state["ge"] = ge
    st.session_state["clf"] = clf
    st.session_state["template_row"] = raw_df.iloc[0].to_dict()

    st.balloons()

    # ===================================================================
    # RESULTS DASHBOARD
    # ===================================================================
    st.markdown("---")
    st.markdown("## :material/bar_chart: Evaluation Results")

    # ---- Metric cards -----------------------------------------------------
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{results['precision'] * 100:.2f}%")
    m3.metric("Recall", f"{results['recall'] * 100:.2f}%")
    m4.metric("F1-Score", f"{results['f1'] * 100:.2f}%")
    m5.metric("ROC-AUC", f"{results['roc_auc'] * 100:.2f}%")
    m6.metric("PR-AUC", f"{results['pr_auc'] * 100:.2f}%")

    st.markdown(f"*Training time: **{train_time:.1f}s** &nbsp;|&nbsp; "
                f"CV F1 Mean: **{results['cv_f1_mean'] * 100:.2f}%** "
                f"(±{results['cv_f1_scores'].std() * 100:.2f}%)*")

    # ---- Classification report --------------------------------------------
    with st.expander(":material/assignment: Full Classification Report", expanded=False):
        st.code(results["classification_report"], language="text")

    # ---- Visualisation columns --------------------------------------------
    st.markdown("### :material/show_chart: Visualisations")
    viz_left, viz_right = st.columns(2)

    # — Confusion Matrix Heatmap —
    with viz_left:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            results["confusion_matrix"],
            annot=True, fmt=",d",
            cmap="rocket_r",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            linewidths=0.5, linecolor="#30363d",
            cbar_kws={"shrink": 0.8},
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted", fontsize=11)
        ax_cm.set_ylabel("Actual", fontsize=11)
        ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="bold", color="#c9d1d9")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

    # — ROC Curve —
    with viz_right:
        st.markdown("#### ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(results["fpr"], results["tpr"],
                    color="#818cf8", linewidth=2,
                    label=f'AUC = {results["roc_auc"] * 100:.2f}%')
        ax_roc.fill_between(results["fpr"], results["tpr"], alpha=0.15, color="#818cf8")
        ax_roc.plot([0, 1], [0, 1], "--", color="#6b7280", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.set_title("ROC Curve", fontsize=13, fontweight="bold", color="#c9d1d9")
        ax_roc.legend(loc="lower right", fontsize=10)
        fig_roc.tight_layout()
        st.pyplot(fig_roc)

    viz_left2, viz_right2 = st.columns(2)

    # — Precision-Recall Curve —
    with viz_left2:
        st.markdown("#### Precision-Recall Curve")
        fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
        ax_pr.plot(results["pr_recall"], results["pr_precision"],
                   color="#c084fc", linewidth=2,
                   label=f'PR-AUC = {results["pr_auc"] * 100:.2f}%')
        ax_pr.fill_between(results["pr_recall"], results["pr_precision"],
                           alpha=0.15, color="#c084fc")
        ax_pr.set_xlabel("Recall", fontsize=11)
        ax_pr.set_ylabel("Precision", fontsize=11)
        ax_pr.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", color="#c9d1d9")
        ax_pr.legend(loc="upper right", fontsize=10)
        fig_pr.tight_layout()
        st.pyplot(fig_pr)

    # — Feature Importance —
    with viz_right2:
        st.markdown("#### Top 15 Feature Importances")
        feat_imp = results["feature_importance"].head(15)
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        bars = ax_fi.barh(
            feat_imp["feature"][::-1],
            feat_imp["importance"][::-1],
            color=plt.cm.cool(np.linspace(0.3, 0.9, len(feat_imp))),
            edgecolor="#30363d",
            linewidth=0.6,
        )
        ax_fi.set_xlabel("Importance", fontsize=11)
        ax_fi.set_title("Feature Importances (Random Forest)",
                        fontsize=13, fontweight="bold", color="#c9d1d9")
        fig_fi.tight_layout()
        st.pyplot(fig_fi)

    # ---- Cross-Validation Box Plot ----------------------------------------
    st.markdown("### :material/sync: Cross-Validation F1 Scores")
    fig_cv, ax_cv = plt.subplots(figsize=(8, 3))
    cv_scores = results["cv_f1_scores"]
    cv_scores_pct = cv_scores * 100
    ax_cv.barh(
        [f"Fold {i+1}" for i in range(len(cv_scores_pct))],
        cv_scores_pct,
        color=["#6366f1" if s >= cv_scores_pct.mean() else "#a78bfa" for s in cv_scores_pct],
        edgecolor="#30363d",
        height=0.55,
    )
    ax_cv.axvline(cv_scores_pct.mean(), color="#f472b6", linestyle="--", linewidth=1.5, label=f"Mean = {cv_scores_pct.mean():.2f}%")
    ax_cv.set_xlabel("F1 Score (%)", fontsize=11)
    ax_cv.legend(fontsize=10)
    ax_cv.set_title("Stratified K-Fold Cross-Validation",
                    fontsize=13, fontweight="bold", color="#c9d1d9")
    fig_cv.tight_layout()
    st.pyplot(fig_cv)

    # ---- Full feature importance table ------------------------------------
    with st.expander(":material/bar_chart: Full Feature Importance Table", expanded=False):
        st.dataframe(
            results["feature_importance"].style.background_gradient(
                cmap="viridis", subset=["importance"]
            ),
            use_container_width=True,
        )

    st.markdown("---")

# ===================================================================
# REAL-TIME PREDICTION INTERFACE
# ===================================================================
if "clf" in st.session_state:
    st.markdown("## :material/online_prediction: Test a Transaction")
    st.markdown("Enter transaction details below to test the trained model. Background data (like location/demographics) will be autofilled based on a template transaction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=10.0)
            test_category = st.selectbox("Category", options=[
                "grocery_pos", "entertainment", "shopping_net", "misc_pos", 
                "shopping_pos", "gas_transport", "fast_food", "health_fitness", 
                "travel", "kids_pets", "misc_net", "personal_care", "food_dining", "grocery_net"
            ])
        with col2:
            test_cc_num = st.text_input("Card Number", value="1234567890123456")
            test_merchant = st.text_input("Merchant Name", value="fraud_merchant")
        with col3:
            test_unix_time = st.number_input("Unix Time", min_value=0, value=int(time.time()), step=3600)

        submitted = st.form_submit_button("Run Prediction", icon=":material/bolt:", use_container_width=True)

    if submitted:
        # Load components from state
        pipeline = st.session_state["pipeline"]
        ge = st.session_state["ge"]
        clf = st.session_state["clf"]
        template = st.session_state["template_row"].copy()

        # Update template with user inputs
        template["amt"] = test_amt
        template["category"] = test_category
        template["cc_num"] = test_cc_num
        template["merchant"] = test_merchant
        template["unix_time"] = test_unix_time

        # Ensure no accidental target variable leakage
        if "is_fraud" in template:
            del template["is_fraud"]

        test_df = pd.DataFrame([template])

        try:
            # 1. Preprocess (fit=False)
            processed_test = pipeline.preprocess(test_df, fit=False)
            
            # 2. Graph features
            enriched_test = ge.extract_features(processed_test)
            
            # 3. Predict
            X_test, _ = clf.prepare_features(enriched_test)
            proba = clf.predict_proba(X_test)[0]
            
            if proba >= 0.5:
                st.error(f"🚨 **FRAUD DETECTED** — {proba * 100:.1f}% Probability", icon=":material/warning:")
            else:
                st.success(f"✅ **LEGITIMATE** — {(1 - proba) * 100:.1f}% Probability (Legit)", icon=":material/check_circle:")
                
            with st.expander("Show Internal Feature Matrix"):
                st.dataframe(X_test)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:0.82rem;opacity:0.7;'>"
    "<span class=\"material-symbols-rounded\" style=\"vertical-align: middle; font-size: 1rem;\">security</span> Fraud Detection System • Graph-Augmented ML Pipeline • "
    "Built with Streamlit, NetworkX & Scikit-learn</p>",
    unsafe_allow_html=True,
)

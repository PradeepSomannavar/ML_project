import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "results" / "plots"
METRICS_PATH = BASE_DIR / "results" / "metrics_summary.csv"

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 28px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    .main-header h1 { font-size: 26px; font-weight: 700; margin: 0 0 6px 0; color: #ffffff; }
    .main-header p { font-size: 14px; color: #c8d6e5; margin: 0; }

    .section-card {
        background: #ffffff;
        padding: 18px 22px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 18px;
    }
    .section-card h2 { font-size: 17px; font-weight: 700; color: #1e293b; margin: 0 0 4px 0; }
    .section-card p { color: #64748b; font-size: 13px; margin: 0; }

    .pipeline-node {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 12px 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        min-height: 56px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 3px 12px rgba(99, 102, 241, 0.25);
    }

    .metric-card {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .metric-card .value { font-size: 24px; font-weight: 700; color: #0f172a; }
    .metric-card .label { font-size: 11px; color: #64748b; margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

    .model-rank {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background: #f8fafc;
        border-radius: 10px;
        margin: 6px 0;
        border-left: 4px solid #6366f1;
    }
    .rank-badge {
        width: 28px; height: 28px;
        background: #6366f1;
        color: white;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 700; flex-shrink: 0;
        margin-right: 12px;
    }
    .model-name { font-weight: 600; font-size: 14px; color: #1e293b; min-width: 140px; }
    .model-acc { font-weight: 700; font-size: 14px; color: #22c55e; margin-left: auto; }

    .step-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #f0f9ff;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 12px;
        color: #1e293b;
        border-left: 3px solid #3b82f6;
        margin: 5px 0;
    }
    .step-num {
        background: #3b82f6;
        color: white;
        width: 22px; height: 22px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700; flex-shrink: 0;
    }

    .plot-section {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .plot-section > div:first-child {
        font-weight: 600;
        font-size: 15px;
        color: #1e293b;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>Network Anomaly Detection Using Machine Learning</h1>
        <p>KDDCup'99 Dataset  |  100,000 Samples  |  5 ML Models  |  Binary Classification (Normal vs Attack)</p>
    </div>
    """, unsafe_allow_html=True)


def show_architecture():
    st.markdown("""
    <div class="section-card">
        <h2>System Architecture</h2>
        <p>End-to-end ML pipeline from raw data to model deployment</p>
    </div>
    """, unsafe_allow_html=True)

    svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture", "ml_pipeline_system_architecture.svg")
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    st.components.v1.html(svg_content, height=900)


def show_pipeline():
    st.markdown("""
    <div class="section-card">
        <h2>Data Preprocessing Pipeline</h2>
        <p>7-step transformation from raw dataset to ML-ready features</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(7)
    steps = [
        ("1", "Load", "100K samples"),
        ("2", "Impute", "Mean fill"),
        ("3", "Dedup", "35.8K rows"),
        ("4", "Encode", "LabelEncoder"),
        ("5", "Scale", "Z-score"),
        ("6", "SMOTE", "Balanced"),
        ("7", "PCA", "41→20"),
    ]

    for i, (num, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f'<div class="pipeline-node"><span style="font-size:18px;">{title}</span><br/><span style="opacity:0.8;font-size:10px;">{desc}</span></div>', unsafe_allow_html=True)
            if i < 6:
                pass

    st.markdown('<div style="text-align:center;margin:4px 0;color:#667eea;font-size:14px;">' + ' ➜  '.join([''] * 7) + '</div>', unsafe_allow_html=True)

    details = [
        "Loaded 100,000 samples with 42 columns (41 features + label)",
        "Mean substitution for numerical columns — 0 missing values",
        "Removed 64,215 duplicate rows (100K → 35,785 unique)",
        "Binary: attack=0, normal=1 | Multi-class: 20 categories",
        "StandardScaler applied to all 41 numerical features",
        "SMOTE: 28,628 → 29,448 (14,724 each class, perfectly balanced)",
        "PCA: 41 → 20 components retaining 95.85% variance"
    ]

    for i, d in enumerate(details):
        st.markdown(f'<div class="step-badge"><div class="step-num">{i+1}</div> {d}</div>', unsafe_allow_html=True)

    st.info("**Final Shapes:** Train = (29,448 × 20) | Test = (7,157 × 20) | 80/20 stratified split")


def show_training():
    st.markdown("""
    <div class="section-card">
        <h2>Model Training & Hyperparameter Tuning</h2>
        <p>RandomizedSearchCV with 3-fold cross-validation, 5 iterations per model</p>
    </div>
    """, unsafe_allow_html=True)

    models_data = [
        {"Model": "Isolation Forest", "Type": "Unsupervised", "Best Params": "N/A (unsupervised)", "Config": "n_estimators=100, contamination=0.1"},
        {"Model": "Naive Bayes", "Type": "Supervised", "Best Params": "var_smoothing = 1e-09", "Config": "GaussianNB (default)"},
        {"Model": "SVM (RBF)", "Type": "Supervised", "Best Params": "C=10, gamma=auto", "Config": "kernel=rbf, probability=True"},
        {"Model": "XGBoost", "Type": "Supervised", "Best Params": "n_estimators=200, max_depth=9, lr=0.01", "Config": "eval_metric=logloss"},
        {"Model": "LightGBM", "Type": "Supervised", "Best Params": "num_leaves=50, n_estimators=100, lr=0.05", "Config": "num_leaves=31 base"}
    ]

    st.dataframe(pd.DataFrame(models_data), use_container_width=True, hide_index=True)

    st.divider()

    training_times = [
        {"Model": "Isolation Forest", "Time": "0.85s", "Note": "Fastest — unsupervised, no labels needed"},
        {"Model": "Naive Bayes", "Time": "0.23s", "Note": "Instant — simple probabilistic model"},
        {"Model": "SVM (RBF)", "Time": "105.55s", "Note": "Slowest — kernel computation on large data"},
        {"Model": "XGBoost", "Time": "6.59s", "Note": "Efficient gradient boosting"},
        {"Model": "LightGBM", "Time": "3.79s", "Note": "Fastest supervised — histogram-based"},
    ]

    st.markdown('<div style="font-weight:600;margin-bottom:8px;">Training Performance</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(training_times), use_container_width=True, hide_index=True)


def show_results():
    metrics = load_metrics()
    if metrics is None:
        st.warning("Metrics file not found.")
        return

    best = metrics.iloc[0]

    st.markdown('<div style="font-weight:700;font-size:18px;margin-bottom:12px;">Performance Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="metric-card"><div class="value">{best["Model"]}</div><div class="label">Best Model</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="value" style="color:#4caf50">{best["Test Accuracy"]:.4f}</div><div class="label">Test Accuracy</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="value" style="color:#2196F3">{best["F1-Score"]:.4f}</div><div class="label">F1-Score</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="value" style="color:#ff9800">{best["AUC-ROC"]:.4f}</div><div class="label">AUC-ROC</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="value">{len(metrics)}</div><div class="label">Models</div></div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div style="font-weight:700;font-size:16px;margin-bottom:12px;">Model Rankings</div>', unsafe_allow_html=True)

    for _, row in metrics.iterrows():
        acc_color = "#4caf50" if row["Test Accuracy"] > 0.95 else "#ff9800" if row["Test Accuracy"] > 0.8 else "#f44336"
        st.markdown(f"""
        <div class="model-rank">
            <div class="rank-badge">{int(row["Rank"])}</div>
            <div class="model-name">{row["Model"]}</div>
            <span style="font-size:12px;color:#888;">Train: {row['Train Accuracy']:.4f}</span>
            <span style="font-size:12px;color:#888;margin:0 12px;">F1: {row['F1-Score']:.4f}</span>
            <span style="font-size:12px;color:#888;">AUC: {row['AUC-ROC']:.4f}</span>
            <div class="model-acc" style="color:{acc_color}">{row['Test Accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-weight:600;margin-bottom:8px;">Accuracy vs F1-Score</div>', unsafe_allow_html=True)
        chart = metrics[["Model", "Test Accuracy", "F1-Score"]].set_index("Model")
        st.bar_chart(chart, use_container_width=True)

    with col_b:
        st.markdown('<div style="font-weight:600;margin-bottom:8px;">95% Confidence Intervals</div>', unsafe_allow_html=True)
        ci_data = metrics[["Model", "Test Accuracy", "CI Lower", "CI Upper"]].copy()
        for c in ["Test Accuracy", "CI Lower", "CI Upper"]:
            ci_data[c] = ci_data[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(ci_data, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown('<div style="font-weight:600;margin-bottom:8px;">Statistical Significance (Paired T-Tests)</div>', unsafe_allow_html=True)
    t_tests = [
        ("Naive Bayes vs SVM", 0.494325),
        ("Naive Bayes vs XGBoost", 0.612916),
        ("Naive Bayes vs LightGBM", 0.492650),
        ("SVM vs XGBoost", 0.438616),
        ("SVM vs LightGBM", 1.000000),
        ("XGBoost vs LightGBM", 0.438616),
    ]
    df_t = pd.DataFrame(t_tests, columns=["Comparison", "p-value"])
    df_t["Result"] = df_t["p-value"].apply(lambda p: "Not Significant (p > 0.05)" if p > 0.05 else "Significant (p < 0.05)")
    st.dataframe(df_t, use_container_width=True, hide_index=True)
    st.info("All top 4 models are statistically equivalent — no significant difference in performance")


def show_plots():
    st.markdown('<div style="font-weight:700;font-size:18px;margin-bottom:16px;">Visualizations</div>', unsafe_allow_html=True)

    if not PLOTS_DIR.exists():
        st.warning("Plots directory not found.")
        return

    plots = list(PLOTS_DIR.glob("*.png"))
    if not plots:
        st.warning("No plots found.")
        return

    titles = {
        "accuracy_vs_f1": "Test Accuracy vs F1-Score Comparison",
        "roc_curves": "ROC Curves — All Models",
        "confusion_matrices": "Confusion Matrices",
        "feature_importance_xgboost": "Top Feature Importance — XGBoost",
        "feature_importance_lightgbm": "Top Feature Importance — LightGBM",
        "pca_variance": "PCA Explained Variance",
        "class_distribution_smote": "Class Distribution Before & After SMOTE",
        "correlation_heatmap": "Feature Correlation Heatmap"
    }

    for p in plots:
        title = titles.get(p.stem, p.stem.replace("_", " ").title())
        st.markdown(f'<div class="plot-section"><div style="font-weight:600;margin-bottom:12px;">{title}</div></div>', unsafe_allow_html=True)
        st.image(str(p), use_container_width=True)


def show_batch():
    st.markdown('<div style="font-weight:700;font-size:18px;margin-bottom:8px;">Batch Prediction</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV file with KDDCup'99 features. All 5 models predict each row simultaneously.")

    try:
        models = {}
        for name, fname in {
            "Isolation Forest": "isolation_forest.joblib",
            "Naive Bayes": "naive_bayes.joblib",
            "SVM": "svm.joblib",
            "XGBoost": "xgboost.joblib",
            "LightGBM": "lightgbm.joblib"
        }.items():
            models[name] = joblib.load(MODELS_DIR / fname)

        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        pca = joblib.load(MODELS_DIR / "pca.joblib")
        le_proto = joblib.load(MODELS_DIR / "le_protocol_type.joblib")
        le_service = joblib.load(MODELS_DIR / "le_service.joblib")
        le_flag = joblib.load(MODELS_DIR / "le_flag.joblib")
        with open(MODELS_DIR / "feature_order.json") as f:
            feature_order = json.load(f)

        encoders = {"protocol_type": le_proto, "service": le_service, "flag": le_flag}
    except FileNotFoundError:
        st.error("Models not found. Run `python main.py` first.")
        return

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)

            for col, le in encoders.items():
                if col in df.columns:
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                    df[col] = le.transform(df[col])

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df_ordered = df[feature_order]
            X = scaler.transform(df_ordered)
            X_pca = pca.transform(X)
            X_raw = X

            preds = {}
            for name, model in models.items():
                if name == "Isolation Forest":
                    preds[name] = np.where(model.predict(X_raw) == -1, "Attack", "Normal")
                else:
                    preds[name] = np.where(model.predict(X_pca) == 0, "Attack", "Normal")

            results_df = pd.DataFrame(preds)

            st.success(f"Predicted {len(results_df)} samples across {len(models)} models")
            st.dataframe(results_df, use_container_width=True)

            total = len(results_df)
            attack_count = (results_df == "Attack").sum().sum()

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="value">{total}</div><div class="label">Total Samples</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="value" style="color:#f44336">{attack_count}</div><div class="label">Attack Detections</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="value" style="color:#4caf50">{total - attack_count}</div><div class="label">Normal</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


def main():
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <h2 style="color:white;">🛡️ Network IDS</h2>
        <p style="color:#889;">KDDCup'99 ML Pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    page = st.sidebar.radio(
        "",
        ["Overview", "Preprocessing", "Training", "Results", "Visualizations", "Batch Prediction"]
    )

    show_header()

    if page == "Overview":
        show_architecture()
    elif page == "Preprocessing":
        show_pipeline()
    elif page == "Training":
        show_training()
    elif page == "Results":
        show_results()
    elif page == "Visualizations":
        show_plots()
    elif page == "Batch Prediction":
        show_batch()


if __name__ == "__main__":
    main()

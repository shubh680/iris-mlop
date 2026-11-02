"""Streamlit Application for Iris Linear Regression Predictor."""

import json
from pathlib import Path
import streamlit as st
from src.predict import IrisPredictor

# Page setup
st.set_page_config(
    page_title="Iris Petal Length Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom modern CSS for a clean, direct, interactive interface
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        max-width: 800px;
        margin: 0 auto;
    }

    .app-header {
        text-align: center;
        padding: 24px 0 16px 0;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0;
    }

    .card {
        background: #181926;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .result-card {
        background: linear-gradient(135deg, #1e1e38 0%, #17182e 100%);
        border: 2px solid #6366f1;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-top: 24px;
        box-shadow: 0 12px 36px rgba(99, 102, 241, 0.2);
    }
    .result-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #a5b4fc;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 4.2rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
        margin: 10px 0;
    }
    .result-unit {
        font-size: 1.6rem;
        font-weight: 600;
        color: #94a3b8;
        margin-left: 6px;
    }
    .species-tag {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 12px;
    }
    .tag-setosa { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .tag-versicolor {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    .tag-virginica { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.4); }

    .calc-breakdown {
        background: rgba(0, 0, 0, 0.25);
        border-radius: 12px;
        padding: 14px;
        margin-top: 18px;
        font-family: monospace;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load and cache the trained linear regression model."""
    return IrisPredictor()


def load_metrics():
    """Load model parameters and metrics."""
    metrics_path = Path(__file__).resolve().parent / "models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


predictor = load_predictor()
metrics = load_metrics()

# App Header
st.markdown(
    """
    <div class="app-header">
        <h1>Iris Petal Length Predictor</h1>
        <p>Enter flower measurements to predict petal length using Linear Regression</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State Defaults
if "sepal_length" not in st.session_state:
    st.session_state.sepal_length = 5.8
if "sepal_width" not in st.session_state:
    st.session_state.sepal_width = 3.0
if "petal_width" not in st.session_state:
    st.session_state.petal_width = 1.3

# Quick Sample Buttons
st.markdown(
    "<p style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 6px;'><b>Quick Presets:</b></p>",
    unsafe_allow_html=True,
)
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

if col_b1.button("Setosa", use_container_width=True):
    st.session_state.sepal_length = 5.0
    st.session_state.sepal_width = 3.4
    st.session_state.petal_width = 0.2
    st.rerun()

if col_b2.button("Versicolor", use_container_width=True):
    st.session_state.sepal_length = 5.9
    st.session_state.sepal_width = 2.8
    st.session_state.petal_width = 1.3
    st.rerun()

if col_b3.button("Virginica", use_container_width=True):
    st.session_state.sepal_length = 6.5
    st.session_state.sepal_width = 3.0
    st.session_state.petal_width = 2.0
    st.rerun()

if col_b4.button("Reset", use_container_width=True):
    st.session_state.sepal_length = 5.8
    st.session_state.sepal_width = 3.0
    st.session_state.petal_width = 1.3
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Main Input Section
with st.container():
    st.markdown("### Enter Floral Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=1.0,
            max_value=15.0,
            value=float(st.session_state.sepal_length),
            step=0.1,
            help="Length of the calyx / sepal leaf in centimeters",
        )

    with col2:
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.sepal_width),
            step=0.1,
            help="Width of the calyx / sepal leaf in centimeters",
        )

    with col3:
        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=8.0,
            value=float(st.session_state.petal_width),
            step=0.1,
            help="Width of the flower petal in centimeters",
        )

# Predict Button & Prediction Computation
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Petal Length", type="primary", use_container_width=True)

# Generate Prediction
predicted_value = predictor.predict(
    sepal_length=sepal_length,
    sepal_width=sepal_width,
    petal_width=petal_width,
)

# Determine closest Iris species cluster
if predicted_value < 2.5:
    species_name = "Iris Setosa"
    species_desc = "Typically small petals (1.0 – 1.9 cm)"
    tag_class = "tag-setosa"
elif predicted_value < 5.0:
    species_name = "Iris Versicolor"
    species_desc = "Medium-sized petals (3.0 – 5.1 cm)"
    tag_class = "tag-versicolor"
else:
    species_name = "Iris Virginica"
    species_desc = "Large floral petals (4.5 – 6.9 cm)"
    tag_class = "tag-virginica"

# Display Prediction Output Card
st.markdown(
    f"""
    <div class="result-card">
        <div class="result-label">Predicted Petal Length</div>
        <div class="result-value">{predicted_value:.2f}<span class="result-unit">cm</span></div>
        <div>
            <span class="species-tag {tag_class}">Most Likely: <b>{species_name}</b> ({species_desc})</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Relative size indicator progress bar
progress_pct = min(max((predicted_value - 1.0) / 6.0, 0.0), 1.0)
st.markdown("<br>", unsafe_allow_html=True)
st.caption(f"Relative Floral Scale: **{predicted_value:.2f} cm** (Iris species range: 1.0 cm – 7.0 cm)")
st.progress(progress_pct)

# Clean, Collapsible Model Details at the bottom
if metrics:
    with st.expander("How the prediction is calculated"):
        coef = metrics["model_parameters"]["coefficients"]
        intercept = metrics["model_parameters"]["intercept"]
        r2 = metrics["metrics"]["test_r2"]
        rmse = metrics["metrics"]["test_rmse"]

        st.markdown(
            f"""
            **Model:** Multiple Linear Regression ($R^2 = {r2:.4f}$, $RMSE = {rmse:.3f}\\text{{ cm}}$)

            $$\\hat{{y}} = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\beta_3 X_3$$

            **Calculation:**
            `{intercept:.3f}` + (`{coef['sepal_length']:.3f}` × {sepal_length}) +
            (`{coef['sepal_width']:.3f}` × {sepal_width}) +
            (`{coef['petal_width']:.3f}` × {petal_width}) = **{predicted_value:.2f} cm**
            """
        )

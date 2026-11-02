"""Streamlit Application for Iris Linear Regression MLOps Suite."""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import IrisPredictor
from src.train import train_model, MODEL_PATH, METRICS_PATH
from src.utils import load_iris_regression_data, FEATURE_NAMES, FEATURE_LABELS, TARGET_NAME

# Page configuration
st.set_page_config(
    page_title="Iris Linear Regression | MLOps Suite",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2b4a 100%);
        padding: 24px 28px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        color: #ffffff;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 8px 0 0 0;
        color: #a0aec0;
        font-size: 1.05rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-purple {
        background-color: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .metric-card {
        background: #181926;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    .prediction-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.1);
        margin-bottom: 24px;
    }
    .prediction-title {
        color: #94a3b8;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .prediction-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
        line-height: 1;
    }
    .prediction-unit {
        font-size: 1.3rem;
        font-weight: 500;
        color: #94a3b8;
        margin-left: 4px;
    }
    .prediction-meta {
        font-size: 0.95rem;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    """Cache model instance to avoid disk reloads."""
    return IrisPredictor()


@st.cache_data
def get_dataset():
    """Cache Iris dataset for exploration."""
    return load_iris_regression_data()


def get_metrics_data():
    """Load cached metrics from JSON or train if missing."""
    if not METRICS_PATH.exists() or not MODEL_PATH.exists():
        train_model()
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


# Load resources
predictor = get_predictor()
df_iris = get_dataset()
metrics_data = get_metrics_data()

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1>🌸 Iris Linear Regression Suite</h1>
        <p>Production MLOps pipeline predicting Iris Petal Length with continuous inference & performance analytics.</p>
        <div class="badge-container">
            <span class="badge badge-blue">⚡ Scikit-Learn Model</span>
            <span class="badge badge-green">✔ CI/CD Automated</span>
            <span class="badge badge-purple">🐳 Docker Containerized</span>
            <span class="badge badge-amber">🚀 Streamlit Cloud</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Parameters & Controls
with st.sidebar:
    st.header("🎛️ Input Features")
    st.caption("Adjust anatomical measurements to estimate petal length:")

    # Quick preset buttons
    st.markdown("**Quick Presets:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_choice = None
    if col_p1.button("Setosa", use_container_width=True):
        st.session_state["sepal_length"] = 5.0
        st.session_state["sepal_width"] = 3.4
        st.session_state["petal_width"] = 0.2
    if col_p2.button("Versicolor", use_container_width=True):
        st.session_state["sepal_length"] = 5.9
        st.session_state["sepal_width"] = 2.8
        st.session_state["petal_width"] = 1.3
    if col_p3.button("Virginica", use_container_width=True):
        st.session_state["sepal_length"] = 6.5
        st.session_state["sepal_width"] = 3.0
        st.session_state["petal_width"] = 2.0

    # Default values from session_state or standard median
    default_sl = st.session_state.get("sepal_length", 5.8)
    default_sw = st.session_state.get("sepal_width", 3.0)
    default_pw = st.session_state.get("petal_width", 1.3)

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=float(default_sl),
        step=0.1,
        help="Calyx leaf length",
    )
    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=float(default_sw),
        step=0.1,
        help="Calyx leaf width",
    )
    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.6,
        value=float(default_pw),
        step=0.1,
        help="True flower petal width",
    )

    st.divider()

    st.markdown("### ⚙️ Pipeline Controls")
    if st.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Retraining Linear Regression model..."):
            new_metrics = train_model()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success(f"Retrained! New Test R²: {new_metrics['metrics']['test_r2']:.4f}")
            st.rerun()

    st.markdown("---")
    st.caption("MLOps Iris Regression • Built with Streamlit & Scikit-Learn")

# Execute Live Prediction
predicted_petal_length = predictor.predict(
    sepal_length=sepal_length,
    sepal_width=sepal_width,
    petal_width=petal_width,
)

# App Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Prediction",
    "📈 Model Evaluation",
    "📊 Dataset Explorer",
    "🚀 Architecture & CI/CD",
])

# ----------------- TAB 1: Live Prediction -----------------
with tab1:
    col_pred, col_details = st.columns([1, 1], gap="large")

    with col_pred:
        # Determine closest typical species based on prediction
        if predicted_petal_length < 2.5:
            likely_species = "Iris Setosa (Typical range: 1.0 - 1.9 cm)"
            species_badge = "🌿 Setosa Cluster"
        elif predicted_petal_length < 5.0:
            likely_species = "Iris Versicolor (Typical range: 3.0 - 5.1 cm)"
            species_badge = "🌺 Versicolor Cluster"
        else:
            likely_species = "Iris Virginica (Typical range: 4.5 - 6.9 cm)"
            species_badge = "🌸 Virginica Cluster"

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-title">Predicted Petal Length</div>
                <div class="prediction-number">{predicted_petal_length:.2f}<span class="prediction-unit">cm</span></div>
                <div class="prediction-meta">
                    <strong>Morphology Association:</strong> {species_badge}<br>
                    <span style="color: #94a3b8; font-size: 0.85rem;">{likely_species}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mathematical Formula Breakdown
        coeffs = metrics_data["model_parameters"]["coefficients"]
        intercept = metrics_data["model_parameters"]["intercept"]

        st.markdown("#### 📐 Regression Equation")
        math_latex = (
            rf"\hat{{y}} = {intercept:.3f} + "
            rf"({coeffs['sepal_length']:.3f} \times \text{{SL}}) + "
            rf"({coeffs['sepal_width']:.3f} \times \text{{SW}}) + "
            rf"({coeffs['petal_width']:.3f} \times \text{{PW}})"
        )
        st.latex(math_latex)

        calc_str = (
            f"**Calculation:** "
            f"`{intercept:.3f}` + "
            f"(`{coeffs['sepal_length']:.3f}` × {sepal_length}) + "
            f"(`{coeffs['sepal_width']:.3f}` × {sepal_width}) + "
            f"(`{coeffs['petal_width']:.3f}` × {petal_width}) = "
            f"**{predicted_petal_length:.2f} cm**"
        )
        st.markdown(calc_str)

    with col_details:
        st.markdown("#### 🗺️ 3D Morphological Space")
        st.caption("Visualizing training samples alongside your live dynamic prediction:")

        # Plot 3D scatter with Plotly
        fig3d = px.scatter_3d(
            df_iris,
            x="petal_width",
            y="sepal_length",
            z="petal_length",
            color="species",
            color_discrete_map={
                "setosa": "#38bdf8",
                "versicolor": "#a855f7",
                "virginica": "#f43f5e",
            },
            opacity=0.6,
            labels={
                "petal_width": "Petal Width (cm)",
                "sepal_length": "Sepal Length (cm)",
                "petal_length": "Petal Length (cm)",
            },
        )

        # Add user's prediction marker
        fig3d.add_trace(
            go.Scatter3d(
                x=[petal_width],
                y=[sepal_length],
                z=[predicted_petal_length],
                mode="markers+text",
                marker=dict(size=10, color="#fbbf24", symbol="diamond", line=dict(color="#ffffff", width=2)),
                name="Your Input",
                text=["Live Prediction"],
                textposition="top center",
            )
        )

        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=20),
            scene=dict(
                xaxis_title="Petal Width (cm)",
                yaxis_title="Sepal Length (cm)",
                zaxis_title="Petal Length (cm)",
                bgcolor="rgba(0,0,0,0)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # 2D Correlation with Interactive Slider
    st.markdown("---")
    st.subheader("🔬 2D Regression View: Petal Width vs Petal Length")
    fig2d = px.scatter(
        df_iris,
        x="petal_width",
        y="petal_length",
        color="species",
        color_discrete_map={
            "setosa": "#38bdf8",
            "versicolor": "#a855f7",
            "virginica": "#f43f5e",
        },
        labels=FEATURE_LABELS,
    )

    # Plot model regression line holding current sepal length & width constant
    pw_vals = [round(0.1 + i * (2.5 / 49), 3) for i in range(50)]
    eval_df = pd.DataFrame(
        {
            "sepal_length": [sepal_length] * 50,
            "sepal_width": [sepal_width] * 50,
            "petal_width": pw_vals,
        }
    )
    predicted_curve = predictor.predict_batch(eval_df)
    fig2d.add_trace(
        go.Scatter(
            x=pw_vals,
            y=predicted_curve,
            mode="lines",
            line=dict(color="#38bdf8", width=2, dash="dash"),
            name=f"Regression Plane (SL={sepal_length}, SW={sepal_width})",
        )
    )

    fig2d.add_trace(
        go.Scatter(
            x=[petal_width],
            y=[predicted_petal_length],
            mode="markers",
            marker=dict(size=14, color="#fbbf24", symbol="star", line=dict(color="#ffffff", width=2)),
            name="Your Prediction",
        )
    )
    fig2d.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        height=380,
    )
    st.plotly_chart(fig2d, use_container_width=True)

# ----------------- TAB 2: Model Evaluation -----------------
with tab2:
    st.subheader("📊 Model Quality Gate & Test Metrics")
    st.caption("Evaluated on an independent 20% test holdout (30 unseen samples):")

    m = metrics_data["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Coefficient of Determination</div>
                <div class="metric-value">{m['test_r2']:.4f}</div>
                <div class="metric-subtext">R² Score (Quality Gate: >0.85)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Root Mean Squared Error</div>
                <div class="metric-value">{m['test_rmse']:.3f} cm</div>
                <div class="metric-subtext">RMSE Error Spread</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value">{m['test_mae']:.3f} cm</div>
                <div class="metric-subtext">Average Prediction Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Training R² Score</div>
                <div class="metric-value">{m['train_r2']:.4f}</div>
                <div class="metric-subtext">Minimal Overfitting Delta</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_coef, col_weights = st.columns([1, 1], gap="large")

    with col_coef:
        st.markdown("#### ⚖️ Feature Importance (Weights / Coefficients)")
        coef_dict = metrics_data["model_parameters"]["coefficients"]
        df_coef = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(k, k) for k in coef_dict.keys()],
                "Coefficient": list(coef_dict.values()),
            }
        ).sort_values(by="Coefficient", ascending=True)

        fig_coef = px.bar(
            df_coef,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="Blues",
            text="Coefficient",
        )
        fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_coef.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_weights:
        st.markdown("#### 🔍 Coefficient Interpretation")
        st.markdown(
            f"""
            - **Petal Width (+{coef_dict['petal_width']:.3f})**: Strongest positive predictor.
              Each 1 cm increase yields ~{coef_dict['petal_width']:.2f} cm increase in petal length.
            - **Sepal Length (+{coef_dict['sepal_length']:.3f})**: Moderate positive contributor
              to overall floral elongation.
            - **Sepal Width ({coef_dict['sepal_width']:.3f})**: Negative coefficient. Broader sepals
              correlate with shorter petals (e.g., Setosa).
            - **Model Intercept ({metrics_data['model_parameters']['intercept']:.3f})**: Base offset constant.
            """
        )

# ----------------- TAB 3: Dataset Explorer -----------------
with tab3:
    st.subheader("📋 Iris Dataset Inspection & Statistics")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        species_filter = st.multiselect(
            "Filter by Species",
            options=df_iris["species"].unique(),
            default=list(df_iris["species"].unique()),
        )
    with col_f2:
        st.metric("Total Filtered Rows", len(df_iris[df_iris["species"].isin(species_filter)]))

    filtered_df = df_iris[df_iris["species"].isin(species_filter)]
    st.dataframe(filtered_df, use_container_width=True, height=260)

    st.markdown("#### 🧬 Correlation Heatmap")
    num_df = df_iris[FEATURE_NAMES + [TARGET_NAME]]
    corr = num_df.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Pearson Correlation Matrix",
    )
    fig_corr.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 4: Architecture & CI/CD -----------------
with tab4:
    st.subheader("🛠️ Production MLOps Architecture & Deployment")

    st.markdown(
        """
        This project is structured according to industry standard **MLOps Best Practices**,
        separating data definitions, model training, artifact persistence, and inference.
        """
    )

    st.markdown("### 1. 🔄 GitHub Actions CI/CD Pipeline")
    st.markdown(
        """
        The `.github/workflows/ci-cd.yml` workflow executes automatically on every `push` and `pull_request` to `main`:
        - **Quality & Linting**: Executes `flake8` to enforce PEP8 standards.
        - **Unit Testing**: Runs `pytest` across all module tests in `tests/`.
        - **Quality Gate Check**: Validates that newly trained model achieves $R^2 > 0.85$ and $RMSE < 1.0$.
        - **Docker Build Validation**: Builds the Docker container to guarantee zero deployment drift.
        """
    )

    st.markdown("### 2. 🐳 Running with Docker")
    st.code(
        """# Build the Docker image
docker build -t iris-linear-regression:latest .

# Run container locally
docker run -p 8501:8501 iris-linear-regression:latest

# Open browser at http://localhost:8501
""",
        language="bash",
    )

    st.markdown("### 3. ☁️ Deploying to Streamlit Community Cloud")
    st.markdown(
        """
        1. Push this repository to your **GitHub** account.
        2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
        3. Click **"New app"**, select your repository, branch (`main`), and set the main file path to: `app.py`.
        4. Click **"Deploy"**! Streamlit automatically reads `requirements.txt` and serves the app.
        """
    )

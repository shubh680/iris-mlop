# 🌸 Iris Linear Regression MLOps Suite

A production-grade, end-to-end Machine Learning project predicting Iris petal length with **Linear Regression** ($R^2 \approx 0.96$). Features an interactive **Streamlit frontend**, automated **GitHub Actions CI/CD pipeline**, **Docker containerization**, and **Streamlit Community Cloud deployment**.

---

## 📌 Project Architecture

```
mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions: Lint, Tests, Quality Gates, Docker build
├── src/
│   ├── __init__.py
│   ├── utils.py               # Data loading, train-test splitting, feature definitions
│   ├── train.py               # Training pipeline, metric calculation, artifact persistence
│   └── predict.py             # Inference pipeline, input validation, and scoring
├── models/
│   ├── linear_regression_model.joblib   # Serialized trained model
│   └── metrics.json           # Model metadata, R², RMSE, MAE, & coefficients
├── tests/
│   ├── __init__.py
│   ├── test_train.py          # Unit tests for training pipeline and performance gates
│   └── test_predict.py        # Unit tests for inference, edge cases, and validations
├── app.py                     # Interactive Streamlit frontend application
├── Dockerfile                 # Multi-stage lightweight production container
├── .dockerignore              # Excludes build caches and virtual envs
├── requirements.txt           # Python dependencies
├── .gitignore                 # Version control exclusions
└── README.md                  # Documentation and deployment manual
```

---

## 🎯 Model & Problem Formulation

- **Dataset**: Fisher's Iris Dataset (Scikit-Learn).
- **Task**: Predict **Petal Length (cm)** based on anatomical measurements.
- **Features**:
  - `sepal_length`: Calyx leaf length (cm)
  - `sepal_width`: Calyx leaf width (cm)
  - `petal_width`: True petal width (cm)
- **Model**: Multiple Linear Regression with analytical least squares estimation.
- **Performance Highlights**:
  - **Test $R^2$**: `0.9603` (High explainability)
  - **Test RMSE**: `0.3606 cm`
  - **Test MAE**: `0.2605 cm`
  - **Equation**: $\text{Petal Length} = -0.262 + 0.723(\text{SL}) - 0.636(\text{SW}) + 1.468(\text{PW})$

---

## 🚀 Quickstart (Local Setup)

### 1. Clone & Set Up Virtual Environment

```bash
cd mlops
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the Model

Train the linear regression model and generate serialized artifacts:

```bash
python -m src.train
```

Outputs:
- `models/linear_regression_model.joblib`
- `models/metrics.json`

### 3. Launch the Streamlit Frontend

```bash
streamlit run app.py
```

The web application opens automatically at `http://localhost:8501`.

---

## 🧪 Automated Testing & Code Quality

Run tests and style linters locally:

```bash
# Run unit tests
pytest -v tests/

# Enforce PEP8 and style guidelines
flake8 src tests app.py --max-line-length=120
```

---

## 🐳 Docker Containerization

Run the entire application in an isolated container:

### Build Container

```bash
docker build -t iris-mlops:latest .
```

### Run Container

```bash
docker run -d -p 8501:8501 --name iris-app iris-mlops:latest
```

Access the application in your browser at: **`http://localhost:8501`**.

---

## 🔄 GitHub Actions CI/CD Pipeline

The included `.github/workflows/ci-cd.yml` workflow triggers on every `push` and `pull_request` to `main`:

1. **Lint & Code Quality**: Validates syntax and enforces PEP8 using `flake8`.
2. **Automated Unit Testing**: Runs test suite with `pytest`.
3. **Model Quality Gate**: Executes training and verifies that test $R^2 \ge 0.85$ and $RMSE \le 1.0$.
4. **Docker Build & Healthcheck**: Verifies Docker container builds cleanly and responds to HTTP health checks on `/_stcore/health`.

---

## ☁️ Deploy to Streamlit Community Cloud (Step-by-Step)

Streamlit Community Cloud provides **free, instantaneous hosting** directly integrated with your GitHub repository.

### Step 1: Initialize Git and Push to GitHub

```bash
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit: Iris Linear Regression MLOps Suite"

# Rename branch to main
git branch -M main

# Add your remote repository (replace with your GitHub username/repo)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

### Step 2: Connect with Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and click **Sign in with GitHub**.
2. Click the **"New app"** button.
3. Select your repository: `<YOUR_USERNAME>/<YOUR_REPO_NAME>`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Deploy!**

Streamlit Cloud installs dependencies from `requirements.txt` and gives you a live public URL (e.g., `https://<YOUR_APP_NAME>.streamlit.app`). Any subsequent `git push` to `main` triggers an automatic redeployment!

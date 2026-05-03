# Network Anomaly Detection Using Machine Learning

> A comprehensive ML-based system for detecting network intrusions and anomalies using the KDDCup'99 dataset. Features a modular training pipeline, model comparison with statistical significance testing, and an interactive Streamlit dashboard for real-time batch prediction.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Pipeline](#pipeline)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements an end-to-end network anomaly detection system that classifies network connections as **Normal** or **Attack** using machine learning. Five distinct models are trained, evaluated, and compared — including both supervised and unsupervised approaches. The system handles the complete data lifecycle from raw dataset ingestion through preprocessing, model training with hyperparameter optimization, statistical evaluation, and interactive visualization.

**Key Features:**
- Automated dataset download and preprocessing pipeline
- SMOTE-based class balancing and PCA dimensionality reduction
- Hyperparameter tuning via RandomizedSearchCV with 3-fold cross-validation
- Statistical significance testing (paired t-tests, confidence intervals)
- Interactive Streamlit dashboard with architecture diagram, results, and batch prediction
- Pre-trained models included — dashboard works immediately without retraining

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION LAYER                            │
│  KDDCup'99 Dataset — 494,021 records → 100,000 sampled              │
│  41 Features + 1 Label (normal / 20 attack types)                   │
└──────────────────────────────────────▼──────────────────────────────┘
                                        │
┌───────────────────────────────────────▼──────────────────────────────┐
│                    PREPROCESSING PIPELINE                             │
│  [1] Load → [2] Missing Values → [3] Duplicates → [4] Encoding       │
│  [5] StandardScaler → [6] SMOTE (balance) → [7] PCA (41→20)         │
└───────────────────────────────────────▼──────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   │                   ▼                  │
┌─────────────────────────────┐ ┌───────▼──────────────────────┐
│      SUPERVISED MODELS      │ │     UNSUPERVISED MODEL       │
│  • Naive Bayes              │ │  • Isolation Forest          │
│  • SVM (RBF Kernel)         │ │    (contamination=0.1)       │
│  • XGBoost                  │ │                              │
│  • LightGBM                 │ │                              │
│  (Hyperparameter tuned)     │ │                              │
└───────────────▼─────────────┘ └───────────▼──────────────────┘
                │                           │
                └───────────────┐   ┌─────────┘
                                ▼   ▼
┌──────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                             │
│  Accuracy | Precision | Recall | F1-Score | AUC-ROC              │
│  Confusion Matrix | Classification Report                        │
│  Paired T-Tests | 95% Confidence Intervals                       │
└───────────────────────────────▼──────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────┐
│                      VISUALIZATION                                │
│  Bar Charts | ROC Curves | Confusion Matrices                    │
│  Feature Importance | PCA Variance | Correlation Heatmap         │
└──────────────────────────────────────────────────────────────────┘
```

The architecture follows a layered design where data flows sequentially through acquisition, preprocessing, model training, evaluation, and visualization stages. The preprocessing pipeline branches into parallel supervised and unsupervised model training, with all predictions converging into a unified evaluation layer for direct model comparison.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | KDDCup'99 (downloaded via sklearn `fetch_kddcup99`) |
| **Total Records** | 494,021 → sampled to 100,000 |
| **After Deduplication** | 35,785 unique records |
| **Features** | 41 TCP/IP connection statistics (numeric + categorical) |
| **Categorical Features** | `protocol_type`, `service`, `flag` |
| **Binary Classes** | Normal (14,724) vs Attack (14,724) after SMOTE |
| **Multi-Classes** | 20 attack categories across 4 types |

### Attack Categories

| Attack Type | Description | Sub-Types |
|---|---|---|
| **DoS** | Denial of Service — overwhelms system resources | back, land, neptune, pod, smurf, teardrop |
| **Probe** | Surveillance and port scanning | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster |
| **R2L** | Remote to Local — unauthorized remote access | buffer_overflow, loadmodule, perl, rootkit |
| **U2R** | User to Root — privilege escalation | ipsweep, nmap, portsweep, satan |

---

## Models

Five models are trained and compared — four supervised classifiers and one unsupervised anomaly detector:

| Rank | Model | Type | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| 🥇 | **LightGBM** | Supervised | 99.79% | 0.9980 | 0.9999 |
| 🥈 | **SVM (RBF)** | Supervised | 99.76% | 0.9977 | 0.9999 |
| 🥉 | **XGBoost** | Supervised | 99.69% | 0.9970 | 0.9999 |
| 4 | **Naive Bayes** | Supervised | 96.40% | 0.9650 | 0.9960 |
| 5 | **Isolation Forest** | Unsupervised | 50.68% | 0.1937 | 0.5003 |

### Hyperparameter Tuning

| Model | Parameters Tuned | Best Configuration |
|---|---|---|
| **LightGBM** | `num_leaves`, `n_estimators`, `learning_rate` | 50 / 100 / 0.05 |
| **XGBoost** | `max_depth`, `n_estimators`, `learning_rate` | 9 / 200 / 0.01 |
| **SVM (RBF)** | `C`, `gamma` | 10 / auto |
| **Naive Bayes** | `var_smoothing` | 1e-09 |
| **Isolation Forest** | N/A (unsupervised) | `n_estimators=100`, `contamination=0.1` |

---

## Pipeline

### Preprocessing Steps

| Step | Operation | Result |
|---|---|---|
| 1 | **Load** | 100,000 samples × 42 columns |
| 2 | **Impute** | Missing values filled via mean substitution |
| 3 | **Deduplicate** | 100,000 → 35,785 unique rows |
| 4 | **Encode** | LabelEncoder on `protocol_type`, `service`, `flag` |
| 5 | **Scale** | StandardScaler (Z-score normalization) |
| 6 | **SMOTE** | Class balancing: 14,724 per class (29,448 total) |
| 7 | **PCA** | Dimensionality reduction: 41 → 20 components (95.85% variance retained) |

### Training Configuration

- **Split:** 80/20 stratified train-test split
- **Cross-Validation:** 3-fold
- **Hyperparameter Search:** RandomizedSearchCV (5 iterations per model)
- **Final Training Shapes:** Train = (29,448 × 20) | Test = (7,157 × 20)

---

## Results

### Statistical Significance

Paired t-tests between all supervised models show no statistically significant difference (p > 0.05), indicating that LightGBM, SVM, and XGBoost are statistically equivalent for this dataset:

| Comparison | p-value | Result |
|---|---|---|
| Naive Bayes vs SVM | 0.4943 | Not Significant |
| Naive Bayes vs XGBoost | 0.6129 | Not Significant |
| Naive Bayes vs LightGBM | 0.4927 | Not Significant |
| SVM vs XGBoost | 0.4386 | Not Significant |
| SVM vs LightGBM | 1.0000 | Not Significant |
| XGBoost vs LightGBM | 0.4386 | Not Significant |

### Visualizations

8 charts are generated in `results/plots/`:
- **Accuracy vs F1-Score** — Bar chart comparison
- **ROC Curves** — Per-model AUC-ROC analysis
- **Confusion Matrices** — TP/FP/TN/FN breakdown
- **Feature Importance (XGBoost)** — Top predictors
- **Feature Importance (LightGBM)** — Top predictors
- **PCA Variance Explained** — Component-wise variance
- **Class Distribution (SMOTE)** — Before/after balancing
- **Correlation Heatmap** — Feature correlation matrix

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

1. Clone the repository:
```bash
git clone https://github.com/PradeepSomannavar/ML_project.git
cd ML_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

> **Note:** Pre-trained models are included in this repository. The dashboard works immediately after installing dependencies — no retraining required.

---

## Usage

### Option 1: Dashboard (Recommended)
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

The dashboard provides six sections:
- **Home** — Project overview with system architecture diagram
- **Pipeline Architecture** — Visual SVG-based architecture diagram
- **Preprocessing** — Detailed 7-step data transformation pipeline
- **Training** — Model configurations and hyperparameter details
- **Results** — Metrics table, statistical tests, and confidence intervals
- **Visualizations** — All 8 generated charts and plots
- **Batch Prediction** — Upload CSV files for model inference

### Option 2: Train from Scratch
Re-run the full training pipeline (overwrites pre-trained models):
```bash
python main.py
```

This executes all steps: data download → preprocessing → training → evaluation → visualization.

---

## Project Structure

```
ML_project/
├── app.py                          # Streamlit dashboard application
├── main.py                         # Full pipeline entry point
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── architecture/
│   └── ml_pipeline_system_architecture.svg   # System architecture diagram
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Data cleaning, SMOTE, PCA, encoding
│   ├── models.py                   # Model definitions (5 models)
│   ├── train.py                    # Training with hyperparameter tuning
│   ├── evaluate.py                 # Metrics, t-tests, confidence intervals
│   └── visualize.py                # Plot generation (8 charts)
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory data analysis
│
├── models/                         # Pre-trained models + artifacts
│   ├── lightgbm.joblib
│   ├── svm.joblib
│   ├── xgboost.joblib
│   ├── naive_bayes.joblib
│   ├── isolation_forest.joblib
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── smote.joblib
│   ├── le_protocol_type.joblib
│   ├── le_service.joblib
│   ├── le_flag.joblib
│   ├── feature_order.json
│   ├── feature_cols.json
│   └── label_info.json
│
├── results/
│   ├── metrics_summary.csv         # Evaluation metrics comparison
│   └── plots/                      # 8 visualization charts (PNG)
│
└── data/                           # Dataset auto-downloaded on first run
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Class Balancing** | imbalanced-learn (SMOTE) |
| **Statistical Analysis** | SciPy (t-tests, confidence intervals) |
| **Visualization** | matplotlib, seaborn |
| **Dashboard** | Streamlit |
| **Model Serialization** | joblib |
| **Dataset** | KDDCup'99 (via sklearn) |

---

## License

This project is for educational purposes. Dataset: KDDCup'99.

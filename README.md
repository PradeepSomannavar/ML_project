# Network Anomaly Detection Using Machine Learning

## Project Overview
Build a complete Network Anomaly Detection system using Machine Learning on the **KDDCup'99** dataset. The system trains, evaluates, and compares 5 ML models to classify network traffic as **Normal** or **Attack**.

## Dataset
- **Source:** KDDCup'99 (via sklearn `fetch_kddcup99`)
- **Features:** 41 TCP/IP connection statistics
- **Classes:** Binary (Normal vs Attack) + 20 multi-class attack types
- **Attacks:** DoS, Probe, R2L, U2R

## Project Structure
```
network-anomaly-project/
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Data cleaning, SMOTE, PCA, encoding
│   ├── models.py           # Model definitions (5 models)
│   ├── train.py            # Training + hyperparameter tuning
│   ├── evaluate.py         # Metrics, t-tests, confidence intervals
│   └── visualize.py        # Plot generation
├── notebooks/
│   └── EDA.ipynb           # Exploratory data analysis
├── results/
│   ├── metrics_summary.csv # Model comparison table
│   └── plots/              # 8 visualization charts
├── models/                 # Trained models + artifacts
├── data/                   # (dataset auto-downloaded)
├── main.py                 # Run full pipeline
├── app.py                  # Streamlit dashboard UI
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models (Full Pipeline)
Runs preprocessing, trains 5 models with hyperparameter tuning, evaluates, and generates plots.
```bash
python main.py
```

### 2. Launch Dashboard UI
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

## Models

| Model | Type | Test Accuracy | F1-Score |
|-------|------|--------------|----------|
| LightGBM | Supervised | 99.79% | 0.9980 |
| SVM (RBF) | Supervised | 99.76% | 0.9977 |
| XGBoost | Supervised | 99.69% | 0.9970 |
| Naive Bayes | Supervised | 96.40% | 0.9650 |
| Isolation Forest | Unsupervised | 50.68% | 0.1937 |

## Pipeline Steps
1. **Load** → KDDCup'99 dataset (100K samples)
2. **Impute** → Missing values (mean substitution)
3. **Dedup** → Remove duplicates (100K → 35.8K)
4. **Encode** → LabelEncoder for categorical features
5. **Scale** → StandardScaler (Z-score normalization)
6. **SMOTE** → Balance classes (14,724 each)
7. **PCA** → Dimensionality reduction (41 → 20 features, 95.85% variance)
8. **Train** → 5 models with RandomizedSearchCV (3-fold)
9. **Evaluate** → Accuracy, Precision, Recall, F1, AUC-ROC, T-tests
10. **Visualize** → Bar charts, ROC curves, confusion matrices, heatmaps

## Results
All metrics saved in `results/metrics_summary.csv` and plots in `results/plots/`.

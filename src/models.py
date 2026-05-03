import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
MODELS_DIR = "models"


def get_isolation_forest():
    return {
        "name": "Isolation Forest",
        "model": IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=RANDOM_STATE
        ),
        "unsupervised": True,
        "param_dist": None
    }


def get_naive_bayes():
    return {
        "name": "Naive Bayes",
        "model": GaussianNB(),
        "unsupervised": False,
        "param_dist": {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    }


def get_svm():
    return {
        "name": "SVM",
        "model": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE
        ),
        "unsupervised": False,
        "param_dist": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"]
        }
    }


def get_xgboost():
    return {
        "name": "XGBoost",
        "model": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
        "unsupervised": False,
        "param_dist": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1]
        }
    }


def get_lightgbm():
    return {
        "name": "LightGBM",
        "model": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        "unsupervised": False,
        "param_dist": {
            "n_estimators": [50, 100, 200],
            "num_leaves": [20, 31, 50],
            "learning_rate": [0.05, 0.1]
        }
    }


def get_all_models():
    return [
        get_isolation_forest(),
        get_naive_bayes(),
        get_svm(),
        get_xgboost(),
        get_lightgbm()
    ]


def save_model(model, name, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.joblib")
    joblib.dump(model, path)
    print(f"  Saved model: {path}")
    return path


def load_model(name, models_dir=MODELS_DIR):
    path = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.joblib")
    return joblib.load(path)

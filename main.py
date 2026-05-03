import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess_pipeline
from src.train import train_all_models
from src.evaluate import evaluate_all_models, paired_t_test
from src.visualize import (
    plot_accuracy_vs_f1, plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_pca_variance, plot_class_distribution,
    plot_correlation_heatmap
)

SAMPLE_SIZE = 100000
TUNE_HYPERPARAMS = True


def save_artifacts(artifacts, label_counts):
    joblib.dump(artifacts["scaler"], "models/scaler.joblib")
    joblib.dump(artifacts["pca"], "models/pca.joblib")
    joblib.dump(artifacts["smote"], "models/smote.joblib")

    label_encoders = artifacts["label_encoders"]
    joblib.dump(label_encoders["protocol_type"], "models/le_protocol_type.joblib")
    joblib.dump(label_encoders["service"], "models/le_service.joblib")
    joblib.dump(label_encoders["flag"], "models/le_flag.joblib")

    feature_order = artifacts["feature_cols"]
    with open("models/feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)

    print("  Saved preprocessing artifacts to models/")


def main():
    print("\n" + "#" * 70)
    print("#  NETWORK ANOMALY DETECTION USING MACHINE LEARNING")
    print("#  Dataset: KDDCup'99 | Sample: 100,000 | Models: 5")
    print("#" * 70 + "\n")

    data_dir = "data"
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # STEP 1: Preprocessing
    data_splits, artifacts, df_raw = preprocess_pipeline(data_dir, sample_size=SAMPLE_SIZE)

    label_counts = dict(zip(*np.unique(data_splits["y_train_bin"], return_counts=True)))
    save_artifacts(artifacts, label_counts)

    y_train_before = data_splits["y_train_bin"]
    y_train_after = data_splits["y_train_res"]
    y_test = data_splits["y_test_bin"]

    # STEP 2: Train all models
    results = train_all_models(data_splits, tune_hyperparams=TUNE_HYPERPARAMS)

    # STEP 3: Evaluate
    df_metrics, all_metrics = evaluate_all_models(results, y_test)

    # STEP 4: Paired t-tests
    t_test_results = paired_t_test(results)

    # STEP 5: Save metrics to CSV
    df_metrics.to_csv("results/metrics_summary.csv", index=False)
    print(f"\nMetrics saved to results/metrics_summary.csv")

    # STEP 6: Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_accuracy_vs_f1(df_metrics)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)

    feature_names = artifacts["feature_cols"]
    for r in results:
        if r["model_name"] in ["XGBoost", "LightGBM"]:
            try:
                n_features = r["model"].n_features_in_
                fn = feature_names[:n_features] if n_features <= len(feature_names) else None
                plot_feature_importance(r["model"], r["model_name"], fn)
            except Exception:
                pass

    plot_pca_variance(artifacts["pca"])
    plot_class_distribution(y_train_before, y_train_after)
    plot_correlation_heatmap(df_raw)

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE")
    print("#" * 70)
    print(f"\nResults saved in:")
    print(f"  - results/metrics_summary.csv")
    print(f"  - results/plots/ (all visualizations)")
    print(f"  - models/ (trained models + artifacts)")
    print()


if __name__ == "__main__":
    main()

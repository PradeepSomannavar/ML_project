import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = None

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": cm
    }


def compute_confidence_interval(acc, n, confidence=0.95):
    z = 1.96 if confidence == 0.95 else 2.576
    se = np.sqrt(acc * (1 - acc) / n)
    margin = z * se
    return {
        "lower": acc - margin,
        "upper": acc + margin,
        "margin": margin
    }


def paired_t_test(results):
    print("\n" + "=" * 60)
    print("PAIRED T-TESTS BETWEEN MODELS")
    print("=" * 60)

    supervised_results = [r for r in results if not r["unsupervised"]]
    names = [r["model_name"] for r in supervised_results]

    test_results = []
    for i in range(len(supervised_results)):
        for j in range(i + 1, len(supervised_results)):
            pred_a = supervised_results[i]["test_predictions"]
            pred_b = supervised_results[j]["test_predictions"]

            y_true = np.zeros(len(pred_a))

            correct_a = (pred_a == y_true).astype(float)
            correct_b = (pred_b == y_true).astype(float)

            t_stat, p_value = stats.ttest_rel(correct_a, correct_b)

            test_results.append({
                "model_a": names[i],
                "model_b": names[j],
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            })

            sig_marker = "*" if p_value < 0.05 else " "
            print(f"  {sig_marker} {names[i]} vs {names[j]}: p={p_value:.6f}")

    return test_results


def evaluate_all_models(results, y_test):
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    n_test = len(y_test)
    all_metrics = []

    for r in results:
        name = r["model_name"]
        y_pred = r["test_predictions"]
        y_proba = r.get("test_proba")

        metrics = compute_metrics(y_test, y_pred, y_proba)

        ci = compute_confidence_interval(metrics["accuracy"], n_test)

        print(f"\n--- {name} ---")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} (95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}])")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics["auc_roc"] is not None:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

        all_metrics.append({
            "Model": name,
            "Train Accuracy": r["train_accuracy"],
            "Test Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "AUC-ROC": metrics["auc_roc"] if metrics["auc_roc"] is not None else 0.0,
            "CI Lower": ci["lower"],
            "CI Upper": ci["upper"],
            "Training Time (s)": r["training_time"]
        })

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    df_metrics["Rank"] = range(1, len(df_metrics) + 1)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY TABLE")
    print("=" * 60)
    print(df_metrics[["Rank", "Model", "Test Accuracy", "Train Accuracy", "F1-Score", "Recall", "Precision", "AUC-ROC"]].to_string(index=False))

    return df_metrics, all_metrics

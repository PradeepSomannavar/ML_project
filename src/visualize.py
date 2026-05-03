import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

PLOTS_DIR = "results/plots"
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


def ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_accuracy_vs_f1(all_metrics):
    print("  Generating: Accuracy vs F1-Score Bar Chart...")
    ensure_dir()

    df = all_metrics.copy()
    models = df["Model"].values
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, df["Test Accuracy"], width, label="Test Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, df["F1-Score"], width, label="F1-Score", color="#DD8452")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Accuracy vs F1-Score Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy_vs_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def plot_roc_curves(results, y_test):
    print("  Generating: ROC Curves...")
    ensure_dir()

    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        y_proba = r.get("test_proba")
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"{r['model_name']} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def plot_confusion_matrices(results, y_test):
    print("  Generating: Confusion Matrices...")
    ensure_dir()

    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, r in enumerate(results):
        y_pred = r["test_predictions"]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        axes[i].set_title(f"{r['model_name']}", fontsize=11)
        axes[i].set_xlabel("Predicted", fontsize=10)
        axes[i].set_ylabel("Actual", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def plot_feature_importance(model_obj, name, feature_names=None):
    print(f"  Generating: Feature Importance for {name}...")
    ensure_dir()

    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
    else:
        return

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title(f"Top 20 Feature Importances - {name}", fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color="#4C72B0", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=9)
    plt.xlabel("Relative Importance", fontsize=12)
    plt.tight_layout()

    clean_name = name.replace(" ", "_").lower()
    path = os.path.join(PLOTS_DIR, f"feature_importance_{clean_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def plot_pca_variance(pca):
    print("  Generating: PCA Variance Plot...")
    ensure_dir()

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = len(cumulative)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_, color="#4C72B0", alpha=0.7)
    ax1.set_xlabel("Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title("Explained Variance per Component", fontsize=14)

    ax2.plot(range(1, n_components + 1), cumulative, marker="o", color="#DD8452", linewidth=2)
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95% Threshold")
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title("Cumulative Explained Variance", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pca_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def plot_class_distribution(y_train_before, y_train_after):
    print("  Generating: Class Distribution (Before/After SMOTE)...")
    ensure_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    classes_before = ["Normal", "Attack"]
    counts_before = [np.sum(y_train_before == 0), np.sum(y_train_before == 1)]
    ax1.bar(classes_before, counts_before, color=["#4C72B0", "#DD8452"])
    ax1.set_title("Before SMOTE", fontsize=14)
    ax1.set_ylabel("Sample Count", fontsize=12)
    for i, v in enumerate(counts_before):
        ax1.text(i, v + 1000, f"{v:,}", ha="center", fontsize=10)

    classes_after = ["Normal", "Attack"]
    counts_after = [np.sum(y_train_after == 0), np.sum(y_train_after == 1)]
    ax2.bar(classes_after, counts_after, color=["#4C72B0", "#DD8452"])
    ax2.set_title("After SMOTE", fontsize=14)
    ax2.set_ylabel("Sample Count", fontsize=12)
    for i, v in enumerate(counts_after):
        ax2.text(i, v + 1000, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution_smote.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def plot_correlation_heatmap(df, top_n=20):
    print(f"  Generating: Correlation Heatmap (top {top_n} features)...")
    ensure_dir()

    numeric_df = df.select_dtypes(include=[np.number])

    correlations = numeric_df.corr().abs()
    mean_corr = correlations.mean().sort_values(ascending=False)
    top_features = mean_corr.head(top_n).index.tolist()

    corr_matrix = numeric_df[top_features].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Heatmap - Top {top_n} Features", fontsize=14)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")


def generate_all_visualizations(results, all_metrics, y_train_before, y_train_after, pca, df_raw, feature_names=None):
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_accuracy_vs_f1(all_metrics)
    plot_roc_curves(results, all_metrics["y_test_bin"] if isinstance(all_metrics, dict) else None)
    plot_confusion_matrices(results, all_metrics["y_test_bin"] if isinstance(all_metrics, dict) else None)

    for r in results:
        if r["model_name"] in ["XGBoost", "LightGBM", "Random Forest"]:
            plot_feature_importance(r["model"], r["model_name"], feature_names)

    plot_pca_variance(pca)
    plot_class_distribution(y_train_before, y_train_after)
    plot_correlation_heatmap(df_raw)

    print("\nAll visualizations saved to results/plots/")
    print("=" * 60)

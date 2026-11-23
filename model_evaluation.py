from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAMES = {
    "Decision Tree": "decision_tree",
    "Bagging": "bagging",
    "Random Forest": "random_forest",
    "AdaBoost": "adaboost",
    "XGBoost": "xgboost",
    "Stacking": "stacking",
    "RF + ANN Hybrid": "rf_ann_hybrid",
}

sns.set_theme(style="whitegrid")


def plot_model_comparison():
    df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    missing_models = set(MODEL_NAMES) - set(df["Model"])
    if missing_models:
        raise ValueError(f"Missing model comparison rows: {sorted(missing_models)}")
    plot_df = df.melt(id_vars="Model", value_vars=metrics,
                      var_name="Metric", value_name="Score")
    plt.figure(figsize=(15, 7))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.title("Comparison of Ensemble Learning Models", fontsize=16, fontweight="bold")
    plt.ylabel("Score")
    plt.xlabel("Machine Learning Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Evaluation Metric")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved model_comparison.png")


def load_predictions():
    df = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    expected = ["actual"]
    for key in MODEL_NAMES.values():
        expected.extend([f"{key}_pred", f"{key}_prob"])
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    return df


def plot_roc_curves(df):
    y_true = df["actual"]
    plt.figure(figsize=(10, 8))
    for name, key in MODEL_NAMES.items():
        fpr, tpr, _ = roc_curve(y_true, df[f"{key}_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Ensemble Learning Comparison", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved roc_curves.png")


def plot_confusion_matrices(df):
    y_true = df["actual"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (name, key) in zip(axes.flat, MODEL_NAMES.items()):
        cm = confusion_matrix(y_true, df[f"{key}_pred"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=ax, xticklabels=["No Experience", "Food Insecure"],
                    yticklabels=["No Experience", "Food Insecure"])
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
    for ax in list(axes.flat)[len(MODEL_NAMES):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices of Seven ML Models", fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrices.png")


def main():
    print("\nGenerating model evaluation charts...\n")
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_predictions()
    plot_model_comparison()
    plot_roc_curves(df)
    plot_confusion_matrices(df)
    print("\nAll evaluation charts generated successfully!")


if __name__ == "__main__":
    main()

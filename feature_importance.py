
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Decision Tree": "decision_tree",
}


def calculate_feature_importance():
    for model_name, filename in MODELS.items():
        print(f"\nAnalyzing {model_name}...")

        pipeline = joblib.load(
            MODELS_DIR / f"{filename}.pkl"
        )

        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        importance = classifier.feature_importances_

        if len(feature_names) != len(importance):
            raise ValueError(
                f"Feature mismatch for {model_name}"
            )

        results = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values(
            "Importance", ascending=False
        )

        results.to_csv(
            RESULTS_DIR / f"{filename}_feature_importance.csv",
            index=False,
        )

        top_features = results.head(15).iloc[::-1]

        plt.figure(figsize=(11, 7))
        plt.barh(
            top_features["Feature"],
            top_features["Importance"],
            color="#16866A",
        )

        plt.title(
            f"Top 15 Important Features — {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Feature Importance")
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"{filename}_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(results.head(15).to_string(index=False))
        print(f"Saved {filename} feature importance.")


if __name__ == "__main__":
    calculate_feature_importance()

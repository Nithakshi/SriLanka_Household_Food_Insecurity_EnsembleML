
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FILES = {
    "identification": "mod_a_household_identification.dta",
    "housing": "mod_c_housing.dta",
    "assets": "mod_d_assets.dta",
    "food_security": "mod_l1_food_security_w.dta",
    "debt": "mod_n1_debt.dta",
    "savings": "mod_n2_savings.dta",
}

QUESTIONS = [f"l1_{i:02d}" for i in range(1, 7)]


def load_and_merge():
    merged = None

    for module, filename in FILES.items():
        path = DATA_DIR / filename

        df = pd.read_stata(
            path,
            convert_categoricals=False
        )

        if "hhcode" not in df.columns:
            raise ValueError(f"Missing hhcode in {filename}")

        if df["hhcode"].duplicated().any():
            raise ValueError(f"Duplicate hhcode in {filename}")

        print(f"Loaded {module}: {df.shape}")

        if merged is None:
            merged = df
        else:
            # Keep shared column names unique
            duplicate_columns = (
                set(merged.columns)
                .intersection(df.columns)
                - {"hhcode"}
            )

            df = df.rename(
                columns={
                    col: f"{module}_{col}"
                    for col in duplicate_columns
                }
            )

            merged = merged.merge(
                df,
                on="hhcode",
                how="inner",
                validate="one_to_one"
            )

    print("\nMerged dataset shape:", merged.shape)
    return merged


def create_target(df):
    missing_questions = [
        q for q in QUESTIONS if q not in df.columns
    ]

    if missing_questions:
        raise ValueError(
            f"Missing food security questions: {missing_questions}"
        )

    responses = df[QUESTIONS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Only Yes (1) and No (2) are valid responses
    valid = responses.isin([1, 2])

    any_yes = responses.eq(1).any(axis=1)
    all_no = responses.eq(2).all(axis=1)

    # Positive if any Yes; negative only if all No
    df["food_insecure"] = np.select(
        [any_yes, all_no],
        [1, 0],
        default=np.nan
    )

    print("\nTarget distribution (including unknown):")
    print(df["food_insecure"].value_counts(dropna=False))

    # Remove households with undetermined targets
    df = df.dropna(subset=["food_insecure"]).copy()
    df["food_insecure"] = df["food_insecure"].astype(int)

    return df


def prepare_features(df):
    # Do not allow the answers used to define the target
    # to become predictors.
    # Exclude the entire food-security module to avoid
    # direct or indirect outcome leakage.
    food_columns = [
        col for col in df.columns
        if col.startswith("l1_")
    ]

    # Remove identifiers and food-security answers.
    # Keep target separately.
    drop_columns = set(
        ["hhcode"] + food_columns
    )

    # Avoid retaining obvious identifier fields.
    # Additional survey identifiers may need review.
    for col in df.columns:
        lower = col.lower()
        if lower in {"id", "household_id", "hhid"}:
            drop_columns.add(col)

    features = df.drop(
        columns=list(drop_columns),
        errors="ignore"
    )

    # Remove completely empty columns
    features = features.dropna(axis=1, how="all")

    # Remove columns that contain only one observed value
    constant_columns = [
        col for col in features.columns
        if col != "food_insecure"
        and features[col].nunique(dropna=True) <= 1
    ]

    features = features.drop(columns=constant_columns)

    print("\nFinal dataset shape:", features.shape)

    return features


def main():
    print("Starting household data preprocessing...\n")

    df = load_and_merge()
    df = create_target(df)
    df = prepare_features(df)

    output_path = RESULTS_DIR / "cleaned_household_data.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved cleaned dataset to:")
    print(output_path)

    print("\nFinal target distribution:")
    print(df["food_insecure"].value_counts())

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()

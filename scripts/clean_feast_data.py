"""Remove physically impossible measurements from the Feast training data.

Why: tests/test_data_validation.py found petal_width = -0.13 for iris_id 1002
on 2025-09-25. Synthetic daily noise pushed a small setosa value below zero.
A length or width must be > 0, so such rows are removed. Values that are
positive but unusual are NOT removed automatically - those need a human review.

Run from the repo root:  python3 scripts/clean_feast_data.py
Safe to run repeatedly: once the bad rows are gone, it changes nothing."""
from pathlib import Path

import pandas as pd

PATH = Path("iris_feature_store/feature_repo/data/iris_data_adapted_for_feast.csv")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

df = pd.read_csv(PATH)
impossible = (df[FEATURES] <= 0).any(axis=1)

print(f"Dropping {impossible.sum()} of {len(df)} rows with non-positive measurements:")
print(df.loc[impossible, ["event_timestamp", "iris_id", *FEATURES, "species"]].to_string())

df.loc[~impossible].to_csv(PATH, index=False)
print(f"Saved {(~impossible).sum()} rows to {PATH}")

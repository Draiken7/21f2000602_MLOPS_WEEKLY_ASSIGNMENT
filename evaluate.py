"""Evaluate the Week 3 Feast model on two evaluation sets.

holdout  : a 40% stratified split (random_state=42) of the DVC-versioned
           Feast training data - the same settings the Week 3 notebook used.
iris_csv : real Iris flowers from data/iris.csv that the model never saw,
           limited to the species the model was trained on.

Used by tests/test_model_evaluation.py and, in CI, by the CML report.
Run from the repo root:  python3 evaluate.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tests.settings import (
    FEAST_DATA,
    FEATURES,
    IRIS_CSV,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_model():
    return joblib.load(MODEL_PATH)


def holdout_set():
    df = pd.read_csv(FEAST_DATA)
    _, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    return test[FEATURES], test[TARGET]


def iris_csv_set(classes):
    df = pd.read_csv(IRIS_CSV)
    df = df[df[TARGET].isin(classes)]
    return df[FEATURES], df[TARGET]


def compute_metrics(model, X, y):
    pred = model.predict(X)
    macro = {"average": "macro", "zero_division": 0}
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, **macro), 4),
        "recall": round(recall_score(y, pred, **macro), 4),
        "f1": round(f1_score(y, pred, **macro), 4),
        "n_samples": int(len(y)),
    }


def evaluate_all(model=None):
    if model is None:
        model = load_model()
    sets = {"holdout": holdout_set(), "iris_csv": iris_csv_set(model.classes_)}
    return {name: compute_metrics(model, X, y) for name, (X, y) in sets.items()}


if __name__ == "__main__":
    results = evaluate_all()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eval_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

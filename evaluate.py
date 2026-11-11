"""Evaluate the IRIS model fetched from the MLflow Model Registry.

The model is resolved by its registered NAME and VERSION:
  - MODEL_VERSION env var set   -> that exact version
  - otherwise                   -> the version the alias @champion points to

Evaluation sets:
  iris_test : the 40% held-out split of data/iris.csv (same split as train.py)
  feast     : the Feast training data (noisy daily measurements the model never saw)

Run from the repo root (MLflow server must be reachable):
    python3 evaluate.py                    # @champion
    MODEL_VERSION=1 python3 evaluate.py    # a specific version
"""
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tests.settings import (
    FEAST_DATA,
    FEATURES,
    IRIS_CSV,
    MODEL_ALIAS,
    MODEL_NAME,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def resolve_model_version():
    """Return (version, uri) for the model to evaluate, by name + version."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    version = os.getenv("MODEL_VERSION")
    if not version:
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    return str(version), f"models:/{MODEL_NAME}/{version}"


def load_model():
    """Fetch the sklearn model from the registry (not from DVC or a local path)."""
    _, uri = resolve_model_version()
    return mlflow.sklearn.load_model(uri)


def iris_test_set():
    df = pd.read_csv(IRIS_CSV)
    _, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                               random_state=RANDOM_STATE)
    return test[FEATURES], test[TARGET]


def feast_set():
    df = pd.read_csv(FEAST_DATA)
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
    sets = {"iris_test": iris_test_set(), "feast": feast_set()}
    return {name: compute_metrics(model, X, y) for name, (X, y) in sets.items()}


if __name__ == "__main__":
    version, uri = resolve_model_version()
    print(f"Evaluating {uri}  (alias @{MODEL_ALIAS} unless MODEL_VERSION is set)")
    results = evaluate_all(mlflow.sklearn.load_model(uri))
    results["model"] = {"name": MODEL_NAME, "version": version}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eval_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

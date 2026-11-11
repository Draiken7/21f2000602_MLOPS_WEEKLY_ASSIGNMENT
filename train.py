"""Week 5 - Train the IRIS classifier with hyperparameter tuning and MLflow.

For every combination in PARAM_GRID this script trains a DecisionTree and logs
one MLflow run with: hyperparameters, data version, cross-validation and test
metrics, and the trained model. The best run (highest cross-validated F1) is
registered in the MLflow Model Registry and gets the alias "champion".

Run from the repo root (MLflow server must be running):
    (MLFLOW_TRACKING_URI=http://127.0.0.1:5000 is set in ~/.bashrc)
    python3 train.py
"""
import itertools
import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from tests.settings import (
    FEATURES,
    IRIS_CSV,
    MODEL_ALIAS,
    MODEL_NAME,
    MLFLOW_EXPERIMENT,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

# Task 1 - the hyperparameters we vary (5 x 2 x 2 = 20 runs)
PARAM_GRID = {
    "max_depth": [1, 2, 3, 5, None],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 5],
}


# A decision tree stores its nodes in this type. We trained the model ourselves,
# so it is safe to trust it when saving and loading.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def data_version(path):
    """md5 of the DVC-tracked data file, read from its .dvc pointer."""
    pointer = path.parent / (path.name + ".dvc")
    if not pointer.exists():
        return "untracked"
    return yaml.safe_load(pointer.read_text())["outs"][0]["md5"]


def test_metrics(model, X, y):
    pred = model.predict(X)
    macro = {"average": "macro", "zero_division": 0}
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, **macro),
        "test_recall": recall_score(y, pred, **macro),
        "test_f1": f1_score(y, pred, **macro),
    }


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = pd.read_csv(IRIS_CSV)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    dvc_md5 = data_version(IRIS_CSV)

    results = []
    keys = list(PARAM_GRID)
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, values))
        run_name = "dt_" + "_".join(f"{k}={v}" for k, v in params.items())
        with mlflow.start_run(run_name=run_name) as run:
            model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
            cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
            model.fit(X_train, y_train)

            metrics = {
                "cv_f1_mean": cv_f1.mean(),
                "cv_f1_std": cv_f1.std(),
                "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
                "tree_depth": model.get_depth(),
                "n_leaves": model.get_n_leaves(),
                **test_metrics(model, X_test, y_test),
            }

            # Task 2 - parameters, metrics and the model for every run
            mlflow.log_params({**params, "model_type": "DecisionTreeClassifier",
                               "test_size": TEST_SIZE, "random_state": RANDOM_STATE})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"tuning_session": session, "data_path": str(IRIS_CSV.name),
                             "data_dvc_md5": dvc_md5})
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(3),
                # MLflow saves sklearn models with skops (safe format) and asks
                # us to confirm the one non-standard type a decision tree uses.
                skops_trusted_types=TRUSTED_TYPES,
            )
            results.append((metrics["cv_f1_mean"], metrics["test_f1"], -metrics["tree_depth"],
                            run.info.run_id, run_name))
            print(f"{run_name:<45} cv_f1={metrics['cv_f1_mean']:.3f}  test_f1={metrics['test_f1']:.3f}")

    # Pick the best run: highest CV F1, then test F1, then the shallower tree
    best = max(results)
    best_run_id, best_name = best[3], best[4]
    print(f"\nBest run: {best_name} ({best_run_id})")

    version = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)
    client.set_model_version_tag(MODEL_NAME, version.version, "selected_by", "cv_f1_mean")
    client.set_model_version_tag(MODEL_NAME, version.version, "tuning_session", session)
    print(f"Registered {MODEL_NAME} version {version.version} with alias @{MODEL_ALIAS}")


if __name__ == "__main__":
    main()

"""Model evaluation - the model is fetched from the MLflow Model Registry.

Registry checks : the alias resolves to a version whose run logged its
                  hyperparameters, metrics and a model (Tasks 2, 5).
Sanity checks   : feature order, known classes, deterministic predictions,
                  textbook flowers.
Quality gates   : accuracy / precision / recall / F1 >= MIN_SCORES on both
                  evaluation sets."""
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from evaluate import (
    compute_metrics,
    feast_set,
    iris_test_set,
    load_model,
    resolve_model_version,
)
from tests.settings import FEATURES, KNOWN_SPECIES, MIN_SCORES, MODEL_NAME

EVAL_SETS = ["iris_test", "feast"]


@pytest.fixture(scope="session")
def model_version():
    try:
        version, _ = resolve_model_version()
    except Exception as exc:  # server down, model or alias missing
        pytest.fail(f"Could not resolve {MODEL_NAME} in the MLflow registry: {exc}")
    return MlflowClient().get_model_version(MODEL_NAME, version)


@pytest.fixture(scope="session")
def model(model_version):
    return load_model()


@pytest.fixture(scope="session")
def eval_sets():
    return {"iris_test": iris_test_set(), "feast": feast_set()}


@pytest.fixture(scope="session")
def scores(model, eval_sets):
    return {name: compute_metrics(model, X, y) for name, (X, y) in eval_sets.items()}


# ---------- Registry checks -----------------------------------------------------

def test_registered_version_is_ready(model_version):
    assert model_version.status == "READY", f"version {model_version.version} is {model_version.status}"


def test_source_run_logged_hyperparameters_and_metrics(model_version):
    run = MlflowClient().get_run(model_version.run_id)
    for param in ["max_depth", "criterion", "min_samples_leaf"]:
        assert param in run.data.params, f"run did not log hyperparameter {param}"
    for metric in ["cv_f1_mean", "test_accuracy", "test_f1"]:
        assert metric in run.data.metrics, f"run did not log metric {metric}"


# ---------- Sanity checks --------------------------------------------------------

def test_feature_order_matches_feast_schema(model):
    assert list(model.feature_names_in_) == FEATURES


def test_classes_are_known_species(model):
    classes = set(model.classes_)
    assert len(classes) >= 2, f"model knows only {classes}"
    assert classes <= KNOWN_SPECIES, f"unknown classes: {classes - KNOWN_SPECIES}"


@pytest.mark.parametrize("name", EVAL_SETS)
def test_eval_set_not_empty(eval_sets, name):
    assert len(eval_sets[name][1]) > 0, f"evaluation set '{name}' is empty"


def test_predictions_valid_and_deterministic(model, eval_sets):
    X, _ = eval_sets["iris_test"]
    first, second = model.predict(X), model.predict(X)
    assert len(first) == len(X)
    assert set(first) <= set(model.classes_)
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "flower, expected",
    [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),
        ([7.0, 3.2, 4.7, 1.4], "versicolor"),
        ([6.3, 3.3, 6.0, 2.5], "virginica"),
    ],
)
def test_textbook_flowers(model, flower, expected):
    if expected not in model.classes_:
        pytest.skip(f"model was not trained on {expected}")
    pred = model.predict(pd.DataFrame([flower], columns=FEATURES))[0]
    assert pred == expected, f"{flower} predicted as {pred}, expected {expected}"


# ---------- Quality gates ---------------------------------------------------------

@pytest.mark.parametrize("name", EVAL_SETS)
@pytest.mark.parametrize("metric", list(MIN_SCORES))
def test_metric_meets_threshold(scores, name, metric):
    value, minimum = scores[name][metric], MIN_SCORES[metric]
    assert value >= minimum, f"{metric} on '{name}' = {value:.3f} < {minimum}"

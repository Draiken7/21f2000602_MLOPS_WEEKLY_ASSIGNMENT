"""Task 2 - Model evaluation for the Week 3 Feast model.

Sanity checks : the model loads, expects the Feast feature order, predicts
                only species it knows, and gives stable, sensible answers.
Quality gates : accuracy / precision / recall / F1 must stay above the
                thresholds in settings.py on BOTH evaluation sets."""
import numpy as np
import pandas as pd
import pytest

from evaluate import compute_metrics, holdout_set, iris_csv_set, load_model
from tests.settings import FEATURES, KNOWN_SPECIES, MIN_SCORES, MODEL_PATH

EVAL_SETS = ["holdout", "iris_csv"]


@pytest.fixture(scope="session")
def model():
    if not MODEL_PATH.exists():
        pytest.fail(f"{MODEL_PATH} not found - run `dvc pull` first")
    return load_model()


@pytest.fixture(scope="session")
def eval_sets(model):
    return {"holdout": holdout_set(), "iris_csv": iris_csv_set(model.classes_)}


@pytest.fixture(scope="session")
def scores(model, eval_sets):
    return {name: compute_metrics(model, X, y) for name, (X, y) in eval_sets.items()}


# ---------- Sanity checks ------------------------------------------------------

def test_model_can_predict(model):
    assert hasattr(model, "predict"), f"{type(model).__name__} has no predict()"


def test_feature_order_matches_feast_schema(model):
    # Training/serving skew guard: the model must expect exactly the
    # FeatureView's features, in the same order.
    assert list(model.feature_names_in_) == FEATURES


def test_classes_are_known_species(model):
    classes = set(model.classes_)
    assert len(classes) >= 2, f"model knows only {classes}"
    assert classes <= KNOWN_SPECIES, f"unknown classes: {classes - KNOWN_SPECIES}"


@pytest.mark.parametrize("name", EVAL_SETS)
def test_eval_set_not_empty(eval_sets, name):
    X, y = eval_sets[name]
    assert len(y) > 0, f"evaluation set '{name}' is empty"


def test_predictions_valid_and_deterministic(model, eval_sets):
    X, _ = eval_sets["iris_csv"]
    first, second = model.predict(X), model.predict(X)
    assert len(first) == len(X), "one prediction per row expected"
    assert set(first) <= set(model.classes_), "predicted an unknown label"
    assert np.array_equal(first, second), "same input gave different predictions"


@pytest.mark.parametrize(
    "flower, expected",
    [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),      # first setosa in the Iris dataset
        ([7.0, 3.2, 4.7, 1.4], "versicolor"),  # first versicolor in the Iris dataset
    ],
)
def test_textbook_flowers(model, flower, expected):
    pred = model.predict(pd.DataFrame([flower], columns=FEATURES))[0]
    assert pred == expected, f"{flower} predicted as {pred}, expected {expected}"


# ---------- Quality gates -------------------------------------------------------

@pytest.mark.parametrize("name", EVAL_SETS)
@pytest.mark.parametrize("metric", list(MIN_SCORES))
def test_metric_meets_threshold(scores, name, metric):
    value, minimum = scores[name][metric], MIN_SCORES[metric]
    assert value >= minimum, f"{metric} on '{name}' = {value:.3f} < {minimum}"

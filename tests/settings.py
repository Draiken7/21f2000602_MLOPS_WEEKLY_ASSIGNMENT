"""One place for every path, column name and threshold the tests use.
If the pipeline changes (new file name, new column), change it here only."""
from pathlib import Path

# Repo root = the folder above tests/. Using an absolute path means the tests
# work no matter which folder pytest is started from.
ROOT = Path(__file__).resolve().parents[1]

# --- Artifacts produced by Week 3 and versioned with DVC ------------------
FEAST_DATA = ROOT / "iris_feature_store/feature_repo/data/iris_data_adapted_for_feast.csv"
MODEL_PATH = ROOT / "artifacts/iris_feast_model.joblib"

# --- Schema: must match the Feast FeatureView in iris_features.py ---------
ENTITY_KEY = "iris_id"                 # Entity join key
EVENT_TS = "event_timestamp"           # when the measurement was true
CREATED_TS = "created_timestamp"       # when the row was written
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
KNOWN_SPECIES = {"setosa", "versicolor", "virginica"}

# Plausible ranges in cm. Real Iris values: sepal length 4.3-7.9,
# sepal width 2.0-4.4, petal length 1.0-6.9, petal width 0.1-2.5.
# Widened a little so noisy-but-real measurements still pass.
FEATURE_RANGES = {
    "sepal_length": (3.0, 9.0),
    "sepal_width": (1.5, 5.0),
    "petal_length": (0.5, 8.0),
    "petal_width": (0.0, 3.0),
}
MIN_ROWS = 30

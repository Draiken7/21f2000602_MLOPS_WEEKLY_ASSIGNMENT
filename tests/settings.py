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

# --- Model evaluation (Stage 5) -------------------------------------------
# Independent evaluation data: the Week 2 Iris file (real flowers the
# Feast model never saw).
IRIS_CSV = ROOT / "data/iris.csv"

# Same split settings as the Week 3 notebook's train_test_split call.
TEST_SIZE = 0.4
RANDOM_STATE = 42

# Quality gates: CI fails if any score drops below these.
# Week 3 reported accuracy 1.0, so 0.9 allows small noise but not a real drop.
MIN_SCORES = {"accuracy": 0.90, "precision": 0.90, "recall": 0.90, "f1": 0.90}

# --- MLflow (Week 5) ------------------------------------------------------
# Models are tracked and stored in MLflow (server on the VM), not in DVC.
MLFLOW_EXPERIMENT = "iris-dt-tuning"   # experiment that groups the tuning runs
MODEL_NAME = "iris-classifier"         # name in the Model Registry
MODEL_ALIAS = "champion"               # alias that points at the best version

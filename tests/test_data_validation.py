"""Task 1 - Data validation for the Feast training data.

Generic checks: schema, missing values, types, value ranges, labels.
Feast checks:   valid timestamps and unique (entity, timestamp) pairs, which
                Feast's point-in-time joins depend on."""
import pandas as pd
import pytest
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from tests.settings import (
    CREATED_TS,
    ENTITY_KEY,
    EVENT_TS,
    FEATURE_RANGES,
    FEATURES,
    KNOWN_SPECIES,
    MIN_ROWS,
    TARGET,
)

ALL_COLUMNS = [EVENT_TS, ENTITY_KEY, *FEATURES, TARGET, CREATED_TS]


# ---------- Generic checks ---------------------------------------------------

def test_has_enough_rows(feast_df):
    assert len(feast_df) >= MIN_ROWS, f"only {len(feast_df)} rows (< {MIN_ROWS})"


def test_expected_columns_present(feast_df):
    missing = set(ALL_COLUMNS) - set(feast_df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"


def test_no_missing_values(feast_df):
    nulls = feast_df[ALL_COLUMNS].isnull().sum()
    assert nulls.sum() == 0, f"missing values per column:\n{nulls[nulls > 0]}"


@pytest.mark.parametrize("col", FEATURES)
def test_feature_is_numeric(feast_df, col):
    assert is_numeric_dtype(feast_df[col]), f"{col} is {feast_df[col].dtype}, expected a number"


@pytest.mark.parametrize("col", FEATURES)
def test_feature_in_plausible_range(feast_df, col):
    low, high = FEATURE_RANGES[col]
    bad = feast_df[(feast_df[col] < low) | (feast_df[col] > high)]
    assert bad.empty, f"{len(bad)} values of {col} outside [{low}, {high}]:\n{bad[[ENTITY_KEY, col]]}"


def test_labels_are_known_species(feast_df):
    labels = set(feast_df[TARGET].unique())
    assert labels <= KNOWN_SPECIES, f"unknown labels: {labels - KNOWN_SPECIES}"


def test_at_least_two_classes(feast_df):
    n = feast_df[TARGET].nunique()
    assert n >= 2, f"only {n} class(es) - a classifier needs at least 2"


# ---------- Feast-specific checks ---------------------------------------------

def test_entity_key_is_positive_integer(feast_df):
    # Feast Entity is declared with ValueType.INT64
    assert is_integer_dtype(feast_df[ENTITY_KEY]), f"{ENTITY_KEY} is {feast_df[ENTITY_KEY].dtype}"
    assert (feast_df[ENTITY_KEY] > 0).all(), f"non-positive {ENTITY_KEY} found"


@pytest.mark.parametrize("col", [EVENT_TS, CREATED_TS])
def test_timestamps_are_valid_datetimes(feast_df, col):
    assert is_datetime64_any_dtype(feast_df[col]), f"{col} could not be parsed as datetime"


def test_no_event_timestamps_in_future(feast_df):
    future = feast_df[feast_df[EVENT_TS] > pd.Timestamp.now()]
    assert future.empty, f"{len(future)} rows have event_timestamp in the future"


def test_created_not_before_event(feast_df):
    # A row cannot be written before the measurement it records happened.
    bad = feast_df[feast_df[CREATED_TS] < feast_df[EVENT_TS]]
    assert bad.empty, f"{len(bad)} rows written before their event time"


def test_entity_timestamp_pairs_unique(feast_df):
    # Two rows for the same plant at the same instant make point-in-time
    # joins ambiguous: Feast could return either value.
    dupes = feast_df[feast_df.duplicated([ENTITY_KEY, EVENT_TS], keep=False)]
    assert dupes.empty, f"duplicate ({ENTITY_KEY}, {EVENT_TS}) pairs:\n{dupes}"


def test_each_entity_has_one_species(feast_df):
    # A single plant cannot change species over time.
    per_id = feast_df.groupby(ENTITY_KEY)[TARGET].nunique()
    bad = per_id[per_id > 1]
    assert bad.empty, f"plants with conflicting labels: {list(bad.index)}"

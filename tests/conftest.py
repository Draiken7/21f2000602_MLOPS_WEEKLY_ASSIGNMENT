"""Shared fixtures. pytest loads this file automatically; any test can use a
fixture just by naming it as a function parameter."""
import pandas as pd
import pytest

from tests.settings import CREATED_TS, EVENT_TS, FEAST_DATA


@pytest.fixture(scope="session")
def feast_df():
    """The Feast training data, loaded once for the whole test run."""
    if not FEAST_DATA.exists():
        pytest.fail(f"{FEAST_DATA} not found - run `dvc pull` first")
    return pd.read_csv(FEAST_DATA, parse_dates=[EVENT_TS, CREATED_TS])

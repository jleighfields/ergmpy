"""Shared fixtures.

The datasets are the authors' committed CSVs rather than synthetic ones: they
are small, they are the specification, and generating a substitute would test
the generator rather than the model.
"""

from pathlib import Path

import pytest

from ergmpy.choice.predict import ChoiceData, load

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
RECORDED_R = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="session")
def train() -> ChoiceData:
    """The training dataset."""
    return load(str(REFERENCE / "Sampled_data_to_share.csv"))


@pytest.fixture(scope="session")
def test_set() -> ChoiceData:
    """The held-out dataset."""
    return load(str(REFERENCE / "test_data_to_share.csv"))


@pytest.fixture(scope="session")
def recorded_r() -> Path:
    """Directory holding outputs saved from R, so no R is needed to compare."""
    return RECORDED_R

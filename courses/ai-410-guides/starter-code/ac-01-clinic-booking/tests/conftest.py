import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clinic_data  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_clinic():
    """Every test starts from the same clinic data."""
    clinic_data.reset()
    yield
    clinic_data.reset()

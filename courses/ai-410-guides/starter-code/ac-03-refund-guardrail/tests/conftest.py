import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import review_queue  # noqa: E402
import store_data  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    """Every test starts from the same orders and an empty queue."""
    store_data.reset()
    review_queue.reset()
    yield
    store_data.reset()
    review_queue.reset()

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obs  # noqa: E402
import push  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_run():
    """Every test starts with no traces and no pushes."""
    obs.reset()
    push.reset()
    yield
    obs.reset()
    push.reset()

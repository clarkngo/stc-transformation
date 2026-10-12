"""
YOUR tests for your paging rule. Write at least two:
  - one order from the day that must page, and one that must not, and
  - the case closest to your rule that worries you most (you can build
    an obs.Trace by hand for a case the day doesn't contain).

The fixture in conftest.py clears traces and pushes before every test.
"""

import instrumented
import obs
from pantry_data import ORDERS

ORDER = {o["order_id"]: o for o in ORDERS}


def test_replace_me():
    ...

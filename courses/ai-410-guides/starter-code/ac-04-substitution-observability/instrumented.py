"""
YOUR WORK GOES HERE — make Puget Pantry's substitution feature observable.

Three functions. The tests in tests/test_instrumented.py check them;
`python run_day.py` runs a day of orders through them.
"""

import feature
import obs
from pantry_data import CATALOG, CUSTOMERS


def score_substitution(original_sku: str, substitute: dict | None, customer_id: str) -> float:
    """How good one substitution was, from 0.0 to 1.0.

      - no substitute (None)                                   -> 0.0
      - the substitute has an allergen the customer listed     -> 0.0
      - the substitute costs more than 20% above the original  -> 0.5
      - otherwise                                              -> 1.0
    """
    raise NotImplementedError


def run_order(order: dict) -> obs.Trace:
    """Run one order through the feature's four steps, traced. Returns the trace.

    The customer must get exactly what feature.run_order_plain would
    send them: tracing may not change behavior.

    Trace "substitute_order" with attributes order_id and customer_id, containing:
      - span "lookup_stock" with attribute out_of_stock (the list of SKUs)
      - for each out-of-stock SKU, span "choose_substitute" with attributes
        item (the SKU), candidates (how many), chosen (the chosen SKU or None),
        and model_ms (how long the model took)
      - if anything was swapped, span "notify" with attributes message and sent=True
        (no notify span, and no push, if nothing was out of stock)
    plus one score "substitution_quality" per swap, with comment = the original SKU.
    """
    raise NotImplementedError


def should_page(trace: obs.Trace) -> bool:
    """YOUR DECISION (see the challenge spec): does this order wake someone up?

    Return True only for what's worth paging the on-call engineer for,
    right now. Everything else belongs on a dashboard.
    """
    raise NotImplementedError

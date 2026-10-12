"""Reference tests for the answer key's paging rule."""

import instrumented
import obs
from pantry_data import ORDERS

ORDER = {o["order_id"]: o for o in ORDERS}


def test_almond_milk_to_tree_nut_allergy_pages():
    assert instrumented.should_page(instrumented.run_order(ORDER["O-502"])) is True


def test_expensive_bread_alone_does_not_page():
    assert instrumented.should_page(instrumented.run_order(ORDER["O-504"])) is False


def test_slow_model_call_does_not_page():
    trace = instrumented.run_order(ORDER["O-507"])   # 3900 ms peanut butter, no allergy
    assert trace.spans_named("choose_substitute")[0].attrs["model_ms"] == 3900
    assert instrumented.should_page(trace) is False


def test_allergen_chosen_but_push_not_sent_does_not_page():
    # A hand-built trace: the push failed, so the customer wasn't told
    # yet, which is a dashboard problem, not a 3 a.m. one.
    trace = obs.start_trace("substitute_order", order_id="O-999", customer_id="K-11")
    with trace.span("choose_substitute", item="milk-whole") as s:
        s.set(candidates=3, chosen="milk-almond", model_ms=800)
    with trace.span("notify") as s:
        s.set(message="...", sent=False)
    assert instrumented.should_page(trace) is False

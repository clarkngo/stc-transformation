"""
Provided tests — your code must pass all of these. Run:  pytest

Don't edit this file. Your own tests go in test_my_rules.py.
"""

import feature
import instrumented
import obs
import push
from pantry_data import CATALOG, ORDERS

ORDER = {o["order_id"]: o for o in ORDERS}


def sub(sku):
    return {"sku": sku, **CATALOG[sku]}


# --- score_substitution ------------------------------------------------

def test_good_substitute_scores_1():
    assert instrumented.score_substitution("milk-whole", sub("milk-soy"), "K-12") == 1.0      # 3.79 vs 3.49: +9%


def test_just_over_20_percent_is_a_price_jump():
    assert instrumented.score_substitution("milk-whole", sub("milk-oat"), "K-12") == 0.5      # 4.29 vs 3.49: +23%


def test_price_jump_scores_half():
    assert instrumented.score_substitution("bread-white", sub("bread-sourdough"), "K-12") == 0.5


def test_allergen_scores_zero_even_if_cheap():
    assert instrumented.score_substitution("milk-whole", sub("milk-almond"), "K-11") == 0.0   # tree nuts


def test_no_substitute_scores_zero():
    assert instrumented.score_substitution("milk-whole", None, "K-12") == 0.0


# --- run_order: the trace ----------------------------------------------

def test_one_trace_per_order_with_its_ids():
    trace = instrumented.run_order(ORDER["O-502"])
    assert obs.TRACES == [trace]
    assert trace.name == "substitute_order"
    assert trace.attrs["order_id"] == "O-502" and trace.attrs["customer_id"] == "K-11"


def test_spans_in_order_with_their_attributes():
    trace = instrumented.run_order(ORDER["O-508"])   # three items out of stock
    assert [s.name for s in trace.spans] == [
        "lookup_stock", "choose_substitute", "choose_substitute", "choose_substitute", "notify"]
    assert trace.spans[0].attrs["out_of_stock"] == ["bread-white", "milk-whole", "yogurt-greek"]
    first = trace.spans[1].attrs
    assert first["item"] == "bread-white" and first["chosen"] == "bread-sourdough"
    assert first["candidates"] == 2 and first["model_ms"] == 640
    assert trace.spans[-1].attrs["sent"] is True
    assert trace.spans[-1].attrs["message"] == push.SENT[0]["message"]


def test_one_quality_score_per_swap():
    trace = instrumented.run_order(ORDER["O-505"])   # peanut butter and white bread, customer allergic to nuts
    assert [(s["name"], s["comment"], s["value"]) for s in trace.scores] == [
        ("substitution_quality", "pb-creamy", 0.0),
        ("substitution_quality", "bread-white", 0.5),
    ]


def test_nothing_out_of_stock_means_no_notify_and_no_push():
    trace = instrumented.run_order(ORDER["O-503"])
    assert [s.name for s in trace.spans] == ["lookup_stock"]
    assert trace.scores == [] and push.SENT == []


def test_tracing_does_not_change_what_customers_get():
    for order in ORDERS:
        instrumented.run_order(order)
    traced = list(push.SENT)
    push.reset()
    for order in ORDERS:
        feature.run_order_plain(order)
    assert traced == push.SENT


# --- should_page -------------------------------------------------------

def test_should_page_answers_yes_or_no_for_every_order():
    for order in ORDERS:
        assert isinstance(instrumented.should_page(instrumented.run_order(order)), bool)


def test_quiet_order_never_pages():
    assert instrumented.should_page(instrumented.run_order(ORDER["O-503"])) is False

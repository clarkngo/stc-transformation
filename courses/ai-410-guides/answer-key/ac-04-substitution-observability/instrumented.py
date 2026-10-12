"""
Reference solution for Application Challenge 4 (faculty only).

Paging rule here is ONE defensible answer: page only when a customer
was sent a substitute containing one of their own allergens. It's rare,
it can hurt someone, and someone can act now (call the customer, stop
the delivery). Price jumps, removed items, and a slow model call go on
the dashboard. On the recorded day this pages for O-502, O-505, O-506.
"""

import feature
import obs
from pantry_data import CATALOG, CUSTOMERS


def score_substitution(original_sku, substitute, customer_id):
    if substitute is None:
        return 0.0
    if substitute["allergens"] & CUSTOMERS[customer_id]["allergies"]:
        return 0.0
    if substitute["price"] > CATALOG[original_sku]["price"] * 1.2:
        return 0.5
    return 1.0


def run_order(order):
    trace = obs.start_trace("substitute_order", order_id=order["order_id"], customer_id=order["customer_id"])
    with trace.span("lookup_stock") as span:
        out_of_stock = feature.lookup_stock(order)
        span.set(out_of_stock=out_of_stock)

    swaps = []
    for sku in out_of_stock:
        with trace.span("choose_substitute", item=sku) as span:
            candidates = feature.find_candidates(sku)
            chosen, ms = feature.choose_substitute(sku, candidates)
            span.set(candidates=len(candidates), chosen=chosen["sku"] if chosen else None, model_ms=ms)
        swaps.append((sku, chosen))

    if swaps:
        with trace.span("notify") as span:
            message = feature.notify(order["customer_id"], swaps)
            span.set(message=message, sent=True)
        for sku, chosen in swaps:
            trace.score("substitution_quality", score_substitution(sku, chosen, order["customer_id"]), comment=sku)
    return trace


def _allergen_sent(trace):
    allergies = CUSTOMERS[trace.attrs["customer_id"]]["allergies"]
    sent = any(s.attrs.get("sent") for s in trace.spans_named("notify"))
    chosen = [s.attrs["chosen"] for s in trace.spans_named("choose_substitute") if s.attrs.get("chosen")]
    return sent and any(CATALOG[c]["allergens"] & allergies for c in chosen)


def should_page(trace):
    return _allergen_sent(trace)

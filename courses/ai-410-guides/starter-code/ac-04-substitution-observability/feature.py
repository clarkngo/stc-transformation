"""
Puget Pantry's substitution feature — someone else's code, already in
production. When an item in an order is out of stock, it asks the model
for a replacement and pushes a notification to the customer's phone.

It works, and nobody can see inside it. Don't change what it does:
your job is to make it observable.

The four steps are separate functions so you can wrap each one in a span.
"""

import fake_model
import push
from pantry_data import CATALOG, CUSTOMERS


def lookup_stock(order: dict) -> list[str]:
    """SKUs in the order that are out of stock."""
    return [sku for sku in order["skus"] if not CATALOG[sku]["in_stock"]]


def find_candidates(sku: str) -> list[dict]:
    """In-stock products in the same category."""
    category = CATALOG[sku]["category"]
    return [{"sku": s, **p} for s, p in CATALOG.items() if p["category"] == category and p["in_stock"] and s != sku]


def choose_substitute(sku: str, candidates: list[dict]) -> tuple[dict | None, int]:
    """The model call. Returns (chosen candidate or None, milliseconds)."""
    return fake_model.choose(sku, candidates)


def notify(customer_id: str, swaps: list[tuple[str, dict | None]]) -> str:
    """Push one message listing every swap. Returns the message."""
    parts = []
    for sku, sub in swaps:
        old = CATALOG[sku]["name"]
        parts.append(f"{old} -> {sub['name']}" if sub else f"{old} removed (no substitute)")
    message = f"Hi {CUSTOMERS[customer_id]['name']}, we made changes to your order: " + "; ".join(parts)
    push.send(customer_id, message)
    return message


def run_order_plain(order: dict) -> list[tuple[str, dict | None]]:
    """The feature as it runs today, with no tracing."""
    swaps = []
    for sku in lookup_stock(order):
        sub, _ms = choose_substitute(sku, find_candidates(sku))
        swaps.append((sku, sub))
    if swaps:
        notify(order["customer_id"], swaps)
    return swaps

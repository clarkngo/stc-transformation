"""
A recorded stand-in for the model that picks substitutes.

In production this step calls Gemini with the item and the in-stock
candidates. Here it replays what the model chose in a test run, so
every trace is the same every time and no key is needed. Like the real
model, it picks the most popular-looking option and is NOT told about
the customer's allergies or the price.

You don't need to change this file.
"""

# How long the real call took for each original item in the recorded run.
_RECORDED_MS = {"milk-whole": 820, "bread-white": 640, "pb-creamy": 3900, "yogurt-greek": 710}


def choose(item_sku: str, candidates: list[dict]) -> tuple[dict | None, int]:
    """Returns (the chosen candidate or None, how many milliseconds it took)."""
    if not candidates:
        return None, 300
    best = max(candidates, key=lambda c: (c["popularity"], c["sku"]))
    return best, _RECORDED_MS.get(item_sku, 500)

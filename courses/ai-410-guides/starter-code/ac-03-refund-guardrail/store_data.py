"""
Pacific Bean Coffee Co. — orders, kept in memory.

You don't need to change this file. Read it so you know what your
refund tool is checking against.

The store runs on a FIXED clock (see `now()`) so the tests give the
same answer every time. Use `store_data.now()`, never datetime.now().
"""

import copy
from datetime import datetime

STORE_NOW = datetime(2026, 11, 16, 10, 0)  # Monday, 10:00 a.m.

REFUND_WINDOW_DAYS = 30
HARD_LIMIT = 500.00  # the owner: "Never let the bot refund more than $500 on its own."

CUSTOMERS = {
    "C-201": "Dana Whitfield",
    "C-202": "Omar Haddad",
    "C-203": "Lena Park (Northgate Cafe, wholesale account)",
}

# order_id -> order. `refunded` is how much has already been refunded.
_INITIAL_ORDERS = {
    "PB-1001": {"customer_id": "C-201", "placed": "2026-11-10", "total": 38.50, "refunded": 0.00,
                "items": ["Single-origin Ethiopia, 2 bags"]},
    "PB-1002": {"customer_id": "C-201", "placed": "2026-10-01", "total": 64.00, "refunded": 0.00,
                "items": ["Pour-over kit"]},
    "PB-1003": {"customer_id": "C-202", "placed": "2026-11-12", "total": 129.00, "refunded": 25.00,
                "items": ["Burr grinder", "House blend, 1 bag"]},
    "PB-1004": {"customer_id": "C-202", "placed": "2026-11-14", "total": 18.00, "refunded": 0.00,
                "items": ["Cold brew concentrate"]},
    "PB-1005": {"customer_id": "C-203", "placed": "2026-11-03", "total": 1240.00, "refunded": 0.00,
                "items": ["Espresso machine (commercial)", "Wholesale beans, 20 lb"]},
    "PB-1006": {"customer_id": "C-203", "placed": "2026-11-15", "total": 310.00, "refunded": 0.00,
                "items": ["Wholesale beans, 25 lb"]},
}

ORDERS = copy.deepcopy(_INITIAL_ORDERS)


def now() -> datetime:
    """The store's current time. Use this instead of datetime.now()."""
    return STORE_NOW


def reset() -> None:
    """Put every order back the way it started. The tests call this."""
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(_INITIAL_ORDERS))

"""Customer-support tools for Harbor Supply Co. — repaired.

Each fix is marked FIX <n>, matching the failure log in FAILURE_LOG.md.
"""

import re
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "harbor.db"

# Deterministic tracking data, so check_shipment gives a final answer instead of "check again".
TRACKING = {
    1001: {"tracking": "delivered", "last_scan": "2026-09-05 Ballard, WA", "carrier": "UPS"},
    1002: {"tracking": "in_transit", "last_scan": "2026-09-23 Tacoma, WA hub", "carrier": "UPS", "eta": "2026-09-25"},
    1004: {"tracking": "in_transit", "last_scan": "2026-09-22 Kent, WA hub", "carrier": "FedEx", "eta": "2026-09-26"},
    1007: {"tracking": "delivered", "last_scan": "2026-09-21 Tacoma, WA", "carrier": "USPS"},
}


def _db():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def _parse_order_id(raw: str) -> int | None:
    # FIX 1: accept "#1004", "ORD-1004", "order 1004", or 1004 instead of crashing on int("#1004").
    digits = re.findall(r"\d+", str(raw))
    return int(digits[-1]) if digits else None


def get_order(order_id: str) -> dict:
    """Look up an order's status, date, and customer.

    Args:
        order_id: The order number, with or without a "#" or "ORD-" prefix, e.g. "1004" or "#1004".
    """
    oid = _parse_order_id(order_id)
    if oid is None:
        return {"status": "error", "error": f"'{order_id}' doesn't contain an order number. Ask the customer for it."}
    con = _db()
    row = con.execute(
        "SELECT o.id, o.placed, o.status, c.name FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.id = ?", (oid,)).fetchone()
    con.close()
    if not row:
        return {"status": "error", "error": f"No order #{oid}."}
    return {"status": "ok", "order_id": row[0], "placed": row[1], "order_status": row[2], "customer": row[3]}


def check_shipment(order_id: int) -> dict:
    """Get the carrier tracking status for a shipped or delivered order. One call gives the current status.

    Args:
        order_id: The numeric order number.
    """
    # FIX 3: return a definite status. The old version always said "check again", which invited a loop.
    info = TRACKING.get(int(order_id))
    if not info:
        return {"status": "error", "error": f"No tracking for order #{order_id}. It may not have shipped yet."}
    return {"status": "ok", "order_id": int(order_id), **info}


def create_support_ticket(order_id: str, reason: str) -> dict:
    """Hand a request you can't complete yourself (such as a refund) to the human support team.

    Args:
        order_id: The order the request is about.
        reason: One sentence describing what the customer needs.
    """
    # FIX 2: the instruction used to promise an issue_refund tool that never existed.
    # Refunds now go to a human, which is also the safer design for an action that moves money.
    oid = _parse_order_id(order_id)
    return {"status": "ok", "ticket_id": f"T-{oid or 0}-R", "handoff": "Support team replies within 1 business day.",
            "reason": reason}


def list_orders() -> dict:
    """List every order with its date, status, and total, plus the overall count and combined value."""
    con = _db()
    # FIX 4: the old query ended in "LIMIT 5" with no sign that rows were missing,
    # so the agent confidently summed 5 of 8 orders. Return everything, plus totals computed here.
    rows = con.execute(
        "SELECT o.id, o.placed, o.status, ROUND(SUM(oi.qty * p.price), 2) FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id "
        "GROUP BY o.id ORDER BY o.id").fetchall()
    con.close()
    return {"status": "ok", "count": len(rows), "combined_value": round(sum(r[3] for r in rows), 2),
            "orders": [{"order_id": i, "placed": d, "order_status": s, "total": t} for i, d, s, t in rows]}


def search_products(query: str) -> dict:
    """Find products by name or category and return their SKUs.

    Args:
        query: A word from the product name or category, e.g. "radio" or "safety".
    """
    con = _db()
    like = f"%{query}%"
    rows = con.execute("SELECT sku, name FROM products WHERE name LIKE ? OR category LIKE ?", (like, like)).fetchall()
    con.close()
    return {"status": "ok", "products": [{"sku": s, "name": n} for s, n in rows]}


def get_product(sku: str) -> dict:
    """Get a product's name, category, retail price (the price customers pay), and stock.

    Args:
        sku: The product SKU, e.g. "VHF-10".
    """
    con = _db()
    row = con.execute("SELECT sku, name, category, price, stock FROM products WHERE sku = ?", (sku,)).fetchone()
    con.close()
    if not row:
        return {"status": "error", "error": f"No product {sku}."}
    return {"status": "ok", "sku": row[0], "name": row[1], "category": row[2], "retail_price": row[3], "in_stock": row[4]}


# FIX 5: get_price returned the store's wholesale cost under the vague name "price",
# and its description ("Get the price of a product") matched customer questions better
# than get_product's did. It's staff-only data, so the customer-facing agent no longer has it.

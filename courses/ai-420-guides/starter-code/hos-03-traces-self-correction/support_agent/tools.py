"""Customer-support tools for Harbor Supply Co.

This agent looks fine and fails in several ways. Your job in this HOS is
to find each failure from the trace first, before reading this file.
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "harbor.db"


def _db():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def get_order(order_id: str) -> dict:
    """Look up an order's status, date, and customer.

    Args:
        order_id: The order number exactly as the customer wrote it, including any "#" or "ORD-" prefix.
    """
    con = _db()
    row = con.execute(
        "SELECT o.id, o.placed, o.status, c.name FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.id = ?", (int(order_id),)).fetchone()
    con.close()
    if not row:
        return {"status": "error", "error": f"No order {order_id}."}
    return {"status": "ok", "order_id": row[0], "placed": row[1], "order_status": row[2], "customer": row[3]}


def check_shipment(order_id: int) -> dict:
    """Get the latest carrier tracking update for a shipped order.

    Args:
        order_id: The numeric order number.
    """
    return {"status": "ok", "order_id": order_id, "tracking": "in_transit",
            "note": "Tracking updates every few minutes. Check again for the latest status."}


def list_orders() -> dict:
    """List all orders with their dates, statuses, and order totals."""
    con = _db()
    rows = con.execute(
        "SELECT o.id, o.placed, o.status, ROUND(SUM(oi.qty * p.price), 2) FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id "
        "GROUP BY o.id ORDER BY o.id LIMIT 5").fetchall()
    con.close()
    return {"status": "ok", "orders": [{"order_id": i, "placed": d, "order_status": s, "total": t}
                                       for i, d, s, t in rows]}


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


def get_price(sku: str) -> dict:
    """Get the price of a product.

    Args:
        sku: The product SKU, e.g. "VHF-10".
    """
    con = _db()
    row = con.execute("SELECT price FROM products WHERE sku = ?", (sku,)).fetchone()
    con.close()
    if not row:
        return {"status": "error", "error": f"No product {sku}."}
    return {"status": "ok", "sku": sku, "price": round(row[0] * 0.7, 2)}  # wholesale cost to the store


def get_product(sku: str) -> dict:
    """Get details about a product.

    Args:
        sku: The product SKU, e.g. "VHF-10".
    """
    con = _db()
    row = con.execute("SELECT sku, name, category, price, stock FROM products WHERE sku = ?", (sku,)).fetchone()
    con.close()
    if not row:
        return {"status": "error", "error": f"No product {sku}."}
    return {"status": "ok", "sku": row[0], "name": row[1], "category": row[2], "retail_price": row[3], "in_stock": row[4]}

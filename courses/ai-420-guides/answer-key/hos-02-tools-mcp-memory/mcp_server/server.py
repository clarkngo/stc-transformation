"""Harbor Supply Co. tool server, speaking MCP over stdio.

Three custom tools, one of each kind the course description names:
  - get_marine_weather  -> an external API (Open-Meteo: free, no key)
  - fetch_page          -> a web browser (fetch a page, return readable text)
  - query_database      -> a database (read-only SQL against data/harbor.db)
Plus two narrow, typed tools written by hand in Stage 4:
  - order_status, low_stock_report

The ADK agent starts this file as a subprocess (see ops_agent/agent.py).
You can also run it alone to check it starts:  python mcp_server/server.py
NOTE: this is mcp 2.x. Older tutorials use `from mcp.server.fastmcp import FastMCP`;
in 2.x that class is `MCPServer` from `mcp.server.mcpserver`.
"""

import logging
import re
import sqlite3
from pathlib import Path

import httpx
import trafilatura
from mcp.server.mcpserver import MCPServer

DB = Path(__file__).resolve().parent.parent / "data" / "harbor.db"
HEADERS = {"User-Agent": "AI420-course-agent/1.0 (https://github.com/clarkngo/stc-transformation; educational use)"}
MAX_ROWS = 50

logging.getLogger("httpx").setLevel(logging.WARNING)  # keep request logs out of the agent's way

server = MCPServer("harbor-tools")


# ---------- API tool ----------

@server.tool()
def get_marine_weather(place: str) -> dict:
    """Current wind and wave conditions for a coastal place, for advising customers heading out on the water.

    Args:
        place: A city or harbor name, e.g. "Anacortes" or "Port Townsend".
    """
    try:
        geo = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": place, "count": 1}, timeout=15).json()
        if not geo.get("results"):
            return {"status": "error", "error": f"Couldn't find a place called '{place}'."}
        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        wx = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=15, params={
            "latitude": lat, "longitude": lon, "wind_speed_unit": "kn",
            "current": "temperature_2m,wind_speed_10m,wind_gusts_10m"}).json()["current"]
        waves = httpx.get("https://marine-api.open-meteo.com/v1/marine", timeout=15, params={
            "latitude": lat, "longitude": lon, "current": "wave_height"}).json().get("current", {})
        return {"status": "ok", "place": f"{loc['name']}, {loc.get('admin1', '')}",
                "air_temp_c": wx["temperature_2m"], "wind_kn": wx["wind_speed_10m"],
                "gusts_kn": wx["wind_gusts_10m"], "wave_height_m": waves.get("wave_height")}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------- Browser tool ----------

@server.tool()
def fetch_page(url: str, max_chars: int = 4000) -> dict:
    """Open a web page and return its main readable text (menus and ads stripped).

    Use this to read a manufacturer's product page, a regulation, or a news article.
    Long pages are cut to max_chars; `truncated` says whether that happened.

    Args:
        url: Full http(s) URL to open.
        max_chars: Most characters of text to return (default 4000).
    """
    if not url.startswith(("http://", "https://")):
        return {"status": "error", "error": "url must start with http:// or https://"}
    try:
        r = httpx.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
        r.raise_for_status()
        text = trafilatura.extract(r.text) or ""
        if not text:
            return {"status": "error", "error": "Page loaded but had no readable text (it may need JavaScript)."}
        return {"status": "ok", "url": str(r.url), "text": text[:max_chars],
                "truncated": len(text) > max_chars, "total_chars": len(text)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------- Database tool ----------

def _read_only():
    # mode=ro makes SQLite itself refuse writes, whatever the SQL says.
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


@server.tool()
def query_database(sql: str) -> dict:
    """Run one read-only SQL SELECT against the Harbor Supply Co. database.

    Tables:
      customers(id, name, email, city)
      products(id, sku, name, category, price, stock, reorder_level)
      orders(id, customer_id, placed, status)   -- status: processing, shipped, delivered, cancelled, backordered
      order_items(order_id, product_id, qty)
    Returns at most 50 rows; `truncated` is true if there were more.

    Args:
        sql: A single SELECT statement.
    """
    statement = sql.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", statement) or ";" in statement:
        return {"status": "error", "error": "Only a single SELECT statement is allowed."}
    try:
        con = _read_only()
        cur = con.execute(statement)
        columns = [c[0] for c in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)
        con.close()
        return {"status": "ok", "columns": columns, "rows": rows[:MAX_ROWS],
                "row_count": min(len(rows), MAX_ROWS), "truncated": len(rows) > MAX_ROWS}
    except sqlite3.Error as e:
        return {"status": "error", "error": f"SQL error: {e}"}


# ---------- Stage 4: narrow, typed tools (written by hand) ----------

@server.tool()
def order_status(order_id: int) -> dict:
    """Look up one order's status, date, customer, and items. Prefer this over query_database for order questions.

    Args:
        order_id: The numeric order number, e.g. 1004 (customers may write it as "#1004").
    """
    con = _read_only()
    order = con.execute(
        "SELECT o.id, o.placed, o.status, c.name FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = ?",
        (order_id,)).fetchone()
    if not order:
        con.close()
        return {"status": "error", "error": f"No order #{order_id}."}
    items = con.execute(
        "SELECT p.name, oi.qty FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE oi.order_id = ?",
        (order_id,)).fetchall()
    con.close()
    return {"status": "ok", "order_id": order[0], "placed": order[1], "order_status": order[2],
            "customer": order[3], "items": [{"product": n, "qty": q} for n, q in items]}


@server.tool()
def low_stock_report() -> dict:
    """List every product whose stock is at or below its reorder level, most urgent first. Use for restocking questions."""
    con = _read_only()
    rows = con.execute(
        "SELECT sku, name, stock, reorder_level FROM products WHERE stock <= reorder_level "
        "ORDER BY stock - reorder_level").fetchall()
    con.close()
    return {"status": "ok", "count": len(rows),
            "products": [{"sku": s, "name": n, "stock": st, "reorder_level": rl} for s, n, st, rl in rows]}


if __name__ == "__main__":
    server.run()  # stdio transport

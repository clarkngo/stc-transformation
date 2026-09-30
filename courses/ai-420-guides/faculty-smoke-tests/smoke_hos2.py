"""Offline smoke test for AI 420 HOS 2. Run from inside a copy of the HOS folder:  python smoke_hos2.py answer|starter"""
import asyncio, json, os, sys, shutil, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fakellm import check, ScriptedLlm, script, run_agent, tool_results
import fakellm

variant = sys.argv[1]
db = Path("data/harbor.db")
if db.exists():
    db.unlink()


async def mcp_checks():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=["mcp_server/server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = sorted(t.name for t in (await s.list_tools()).tools)
            check("MCP server starts over stdio and lists tools", names, names)
            check("harbor.db auto-built on first start (portfolio Codespace case)", db.exists())

            async def call(name, args):
                res = await s.call_tool(name, args)
                sc = getattr(res, "structuredContent", None) or getattr(res, "structured_content", None)
                if sc:
                    return sc.get("result", sc)
                return json.loads(res.content[0].text)

            w = await call("get_marine_weather", {"place": "Anacortes"})
            check("get_marine_weather live (Open-Meteo)", w.get("status") == "ok", f"wind {w.get('wind_kn')} kn, waves {w.get('wave_height_m')} m")
            if variant == "answer":
                p = await call("fetch_page", {"url": "https://en.wikipedia.org/wiki/Personal_flotation_device"})
                check("fetch_page live (trafilatura)", p.get("status") == "ok" and p.get("total_chars", 0) > 500, f"{p.get('total_chars')} chars")
                check("fetch_page rejects non-http", (await call("fetch_page", {"url": "file:///etc/passwd"}))["status"] == "error")
                q = await call("query_database", {"sql": "SELECT name FROM products WHERE category='safety'"})
                check("query_database SELECT", q.get("status") == "ok" and q.get("row_count") == 3, q.get("rows"))
                for bad in ["DELETE FROM orders", "SELECT 1; DROP TABLE orders", "UPDATE orders SET status='cancelled' WHERE id=1003"]:
                    check(f"query_database refuses: {bad[:30]}", (await call("query_database", {"sql": bad}))["status"] == "error")
                check("query_database bad column -> error dict", (await call("query_database", {"sql": "SELECT nope FROM orders"}))["status"] == "error")
                o = await call("order_status", {"order_id": 1003})
                check("order_status (Stage 4)", o.get("order_status") == "processing" and o.get("items"), o.get("customer"))
                l = await call("low_stock_report", {})
                check("low_stock_report (Stage 4)", l.get("count", 0) >= 3, [x["sku"] for x in l.get("products", [])])
            else:
                check("starter serves only the worked example tool", names == ["get_marine_weather"])


asyncio.run(mcp_checks())

import sqlite3
con = sqlite3.connect(db)
check("DB unchanged after write attempts (order 1003 still processing)",
      con.execute("select status from orders where id=1003").fetchone()[0] == "processing")
con.close()

# ADK agent loads and discovers the MCP tools the way adk web does.
from ops_agent.agent import root_agent


async def adk_checks():
    toolset = [t for t in root_agent.tools if t.__class__.__name__ == "McpToolset"][0]
    tools = await toolset.get_tools()
    check("ADK McpToolset discovers MCP tools", tools, sorted(t.name for t in tools))
    if variant == "answer":
        from ops_agent import memory
        memory.remember_customer_fact("Ana Reyes", "Keeps a 26-foot sailboat named Osprey at Shilshole")
        memory.remember_customer_fact("Marcus Webb", "Prefers phone calls")
        rec = memory.recall_customer_facts("Ana Reyes", "her boat")
        check("long-term memory: save + recall by meaning (Chroma)", any("Osprey" in f for f in rec["facts"]), rec["facts"])
        check("long-term memory: scoped to the customer", not any("phone" in f for f in rec["facts"]))
        root_agent.model = ScriptedLlm(policy=script([
            ("call", "set_current_customer", {"customer_name": "Ana Reyes"}),
            ("call", "order_status", {"order_id": 1004}),
            ("text", "Your order 1004 has shipped."),
        ]))
        events, session = await run_agent(root_agent, "Hi, I'm Ana Reyes. Where's order 1004?")
        res = dict(tool_results(events))
        check("ADK run: short-term memory in session state", session.state.get("current_customer") == "Ana Reyes")
        os_res = res.get("order_status", {})
        text = json.dumps(os_res, default=str)
        check("ADK run: agent -> MCP tool call round trip", "shipped" in text, text[:120])
    await toolset.close()


asyncio.run(adk_checks())
print("\nHOS 2", variant, "->", "ALL PASS" if fakellm.OK else "FAILURES ABOVE")
sys.exit(0 if fakellm.OK else 1)

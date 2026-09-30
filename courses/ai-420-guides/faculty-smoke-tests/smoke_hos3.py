"""Offline smoke test for AI 420 HOS 3: do the planted failures reproduce (starter) and recover (answer)?
Drives the real run_scenarios.run_one with a scripted stand-in model that behaves like a naive model would.
    python smoke_hos3.py answer|starter
"""
import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fakellm import check, ScriptedLlm, script
import fakellm

variant = sys.argv[1]
if Path("data/harbor.db").exists():
    Path("data/harbor.db").unlink()

import run_scenarios
from support_agent.agent import root_agent
check("harbor.db auto-built on import (portfolio Codespace case)", Path("data/harbor.db").exists())
scen = {s["id"]: s for s in json.load(open("scenarios.json"))}
check("6 scenarios", len(scen) == 6)

POLICIES = {
    "s1-order-lookup": [("call", "get_order", {"order_id": "#1004"}), ("text", "Order 1004 has shipped.")],
    "s2-refund": [("call", "issue_refund", {"order_id": "1005"}), ("text", "Your refund has been processed.")],
    "s3-where-is-it": [("call", "check_shipment", {"order_id": 1002})] * 6 + [("text", "It's in transit.")],
    "s4-monthly-total": [("call", "list_orders", {}), ("text", "Summary.")],
    "s5-price-quote": [("call", "search_products", {"query": "radio"}),
                       ("call", "get_price" if variant == "starter" else "get_product", {"sku": "VHF-10"}),
                       ("text", "Quote.")],
    "s6-control": [("call", "search_products", {"query": "safety"}), ("text", "Life jacket, flare kit, fire extinguisher.")],
}
ANSWER_S2 = [("call", "get_order", {"order_id": "1005"}),
             ("call", "create_support_ticket", {"order_id": "1005", "reason": "Refund for cancelled order"}),
             ("text", "I've opened a ticket; support replies within 1 business day.")]


async def run(sid, steps):
    root_agent.model = ScriptedLlm(policy=script(steps))
    return await run_scenarios.run_one(scen[sid], max_calls=12)


def results(r, tool):
    return [s["result"] for s in r["trace"] if s["kind"] == "tool_result" and s["tool"] == tool]


def crashed(r):
    return any(s["kind"] in ("crash", "error") for s in r["trace"])


async def main():
    r1 = await run("s1-order-lookup", POLICIES["s1-order-lookup"])
    if variant == "starter":
        check('s1 reproduces: "#1004" crashes get_order', crashed(r1) or any(x.get("status") == "error" for x in results(r1, "get_order")),
              next((s.get("error") for s in r1["trace"] if s["kind"] in ("crash", "error")), "")[:90])
    else:
        check('s1 fixed: "#1004" parsed', any(x.get("order_id") == 1004 for x in results(r1, "get_order")))

    r2 = await run("s2-refund", POLICIES["s2-refund"])
    if variant == "starter":
        check("s2 reproduces: instruction promises issue_refund, which doesn't exist", crashed(r2) or any("issue_refund" in json.dumps(s, default=str) for s in r2["trace"] if s["kind"] == "tool_result"),
              next((s.get("error") for s in r2["trace"] if s["kind"] in ("crash", "error")), "")[:90])
    else:
        check("s2 guard: a hallucinated tool call doesn't crash the run", not any(s["kind"] == "crash" for s in r2["trace"]),
              next((s.get("error") for s in r2["trace"] if s["kind"] in ("crash", "error")), "handled")[:90])
        r2b = await run("s2-refund", ANSWER_S2)
        check("s2 fixed path: get_order -> create_support_ticket", results(r2b, "create_support_ticket") and results(r2b, "create_support_ticket")[0].get("ticket_id"))

    r3 = await run("s3-where-is-it", POLICIES["s3-where-is-it"])
    shipments = results(r3, "check_shipment")
    if variant == "starter":
        check("s3 reproduces: 'check again' invites a loop (6 identical calls, no definite status)",
              len(shipments) == 6 and all("Check again" in x.get("note", "") for x in shipments))
    else:
        check("s3 fixed: definite tracking status", shipments and shipments[0].get("eta"))
        check("s3 guard: repeat calls blocked after 2", any("already called" in x.get("error", "") for x in shipments))

    r4 = await run("s4-monthly-total", POLICIES["s4-monthly-total"])
    lo = results(r4, "list_orders")[0]
    if variant == "starter":
        total5 = round(sum(o["total"] for o in lo["orders"]), 2)
        check("s4 reproduces: list_orders silently returns 5 of 8 orders", len(lo["orders"]) == 5 and "count" not in lo, f"sum of 5 = ${total5}")
    else:
        check("s4 fixed: all orders + count + combined_value", lo.get("count") == len(lo["orders"]) and lo["count"] > 5, f"{lo.get('count')} orders, ${lo.get('combined_value')}")

    r5 = await run("s5-price-quote", POLICIES["s5-price-quote"])
    if variant == "starter":
        p = results(r5, "get_price")[0]
        check("s5 reproduces: get_price returns wholesale cost as 'price'", p.get("price") == 153.3, f"price={p.get('price')}")
    else:
        p = results(r5, "get_product")[0]
        check("s5 fixed: retail_price quoted, get_price removed", p.get("retail_price") == 219.0 and not any(t.name == "get_price" for t in getattr(root_agent, "tools", []) if hasattr(t, "name")))

    r6 = await run("s6-control", POLICIES["s6-control"])
    names = [x["name"] for x in results(r6, "search_products")[0]["products"]]
    check("s6 control works in both", len(names) == 3, names)

    Path("traces").mkdir(exist_ok=True)
    (Path("traces") / "smoke.json").write_text(json.dumps(r6, indent=2, default=str))
    check("traces/ written as JSON", (Path("traces") / "smoke.json").exists())


asyncio.run(main())
if variant == "answer":
    import subprocess
    out = subprocess.run([sys.executable, "-c", "import tracing"], capture_output=True, text=True,
                         env={**os.environ, "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""})
    check("tracing.py without Langfuse keys: local traces only, no crash", out.returncode == 0 and "local traces only" in out.stdout, out.stdout.strip())
print("\nHOS 3", variant, "->", "ALL PASS" if fakellm.OK else "FAILURES ABOVE")
sys.exit(0 if fakellm.OK else 1)

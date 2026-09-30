"""Offline smoke test for AI 420 HOS 4.  python smoke_hos4.py answer|starter"""
import asyncio, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fakellm import check, ScriptedLlm, script, run_agent, tool_results, final_text
import fakellm
from google.genai import types

variant = sys.argv[1]
if Path("data/harbor.db").exists():
    Path("data/harbor.db").unlink()

from single_agent.agent import root_agent as single
check("harbor.db auto-built on import (portfolio Codespace case)", Path("data/harbor.db").exists())
check("12 evaluation scenarios", len(json.load(open("evals/scenarios.json"))) == 12)
pricing = json.load(open("evals/pricing.json"))
check("pricing.json has input/output prices", {"input_per_million", "output_per_million"} <= set(pricing))


async def single_run():
    single.model = ScriptedLlm(policy=script([("call", "get_order", {"order_id": "#1004"}), ("text", "Order 1004 has shipped.")]))
    ev, _ = await run_agent(single, "Status of #1004?")
    res = dict(tool_results(ev))
    check("single_agent (repaired HOS 3 baseline) runs", res.get("get_order", {}).get("order_status") == "shipped")

asyncio.run(single_run())

if variant == "starter":
    check("team_agent/ and run_eval.py are left for the student", not Path("team_agent").exists() and not Path("evals/run_eval.py").exists())
else:
    sys.path.insert(0, "evals")
    from team_agent.agent import root_agent as team
    import run_eval

    check("team: coordinator + 3 specialists, coordinator has no tools",
          len(team.sub_agents) == 3 and not team.tools, [a.name for a in team.sub_agents])

    async def team_run():
        shared = ScriptedLlm(policy=script([
            ("call", "transfer_to_agent", {"agent_name": "orders_agent"}),
            ("call", "get_order", {"order_id": "1004"}),
            ("text", "Order 1004 shipped on its way."),
        ]))
        for a in [team] + list(team.sub_agents):
            a.model = shared
        ev, _ = await run_agent(team, "Where is order 1004?")
        authors = [e.author for e in ev]
        check("team: coordinator hands off to orders_agent", "orders_agent" in authors, sorted(set(authors)))
        check("team: specialist's tool call works", dict(tool_results(ev)).get("get_order", {}).get("order_status") == "shipped")
        return shared
    asyncio.run(team_run())

    class FakeClient:
        """Stand-in for genai.Client: simulated user says one thing then DONE; judge passes."""
        def __init__(self): self.models = self; self.n = 0
        def generate_content(self, model, contents, config=None):
            self.n += 1
            if config is not None and getattr(config, "response_mime_type", None) == "application/json":
                text = json.dumps({"success": True, "reason": "Agent stated order 1004 shipped."})
            else:
                text = "Hi, what's the status of #1004?" if "(no messages yet)" in contents else "DONE"
            return types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
                usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=50))

    async def harness():
        scen = json.load(open("evals/scenarios.json"))[0]
        results = {}
        for name, agent in (("single", single), ("team", team)):
            steps = ([("call", "transfer_to_agent", {"agent_name": "orders_agent"})] if name == "team" else []) + \
                    [("call", "get_order", {"order_id": "#1004"}), ("text", "Order 1004 has shipped (placed 2026-09-10).")]
            m = ScriptedLlm(policy=script(steps))
            for a in [agent] + list(getattr(agent, "sub_agents", [])):
                a.model = m
            client = FakeClient()
            r = await run_eval.run_scenario(agent, client, scen, max_turns=4, max_calls=12)
            results[name] = r
            check(f"harness ({name}): sim user -> agent -> judge", r["success"] and r["turns"] == 1 and not r["errors"],
                  f"turns={r['turns']} tokens_in={r['tokens_in']} out={r['tokens_out']} sim+judge calls={client.n}")
        return results
    res = asyncio.run(harness())

    # Metric math on known numbers.
    fake = [{"success": True, "tokens_in": 1_000_000, "tokens_out": 100_000, "reply_seconds": [1, 2, 3, 4],
             "errors": [], "eval_overhead_tokens": 10},
            {"success": False, "tokens_in": 0, "tokens_out": 0, "reply_seconds": [10], "errors": ["x"], "eval_overhead_tokens": 5}]
    s = run_eval.summarize(fake, {"input_per_million": 0.30, "output_per_million": 2.50})
    check("metrics: success rate", s["success_rate"] == 0.5)
    check("metrics: cost per task = (1M x $0.30 + 0.1M x $2.50) / 2 = $0.275", s["cost_per_task_usd"] == 0.275, s["cost_per_task_usd"])
    check("metrics: p50 / p95 latency (nearest rank)", s["latency_p50_s"] == 3 and s["latency_p95_s"] == 10, (s["latency_p50_s"], s["latency_p95_s"]))
    check("metrics: crashes counted", s["crashes"] == 1)

    out = Path("evals/results"); out.mkdir(exist_ok=True)
    for name, r in res.items():
        (out / f"{name}.json").write_text(json.dumps({"system": name, "model": "stand-in",
                                                      "summary": run_eval.summarize([r], pricing), "results": [r]}, indent=2))
    rep = subprocess.run([sys.executable, "evals/compare.py"], capture_output=True, text=True)
    check("compare.py prints the side-by-side table", rep.returncode == 0 and "| Success rate |" in rep.stdout, rep.stderr[-120:])
    Path("evals/human_labels.json").write_text(json.dumps({"order-status": True}))
    ag = subprocess.run([sys.executable, "evals/agreement.py", "--system", "team"], capture_output=True, text=True)
    check("agreement.py (Stage 4) reports judge agreement", "Judge agreed with you on 1/1" in ag.stdout, ag.stdout.strip()[:80] or ag.stderr[-120:])

print("\nHOS 4", variant, "->", "ALL PASS" if fakellm.OK else "FAILURES ABOVE")
sys.exit(0 if fakellm.OK else 1)

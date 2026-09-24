"""Agent-on-agent evaluation: a simulated customer talks to the agent system, a judge grades each conversation.

    python evals/run_eval.py --system single --limit 4     # try a few first
    python evals/run_eval.py --system team
    python evals/compare.py                                 # side-by-side report

Three roles, three model calls per turn or so:
  simulated user  -> plays the persona in scenarios.json, pursuing its goal (SIM_MODEL)
  agent system    -> the thing being tested (single_agent or team_agent)
  judge           -> reads the transcript, grades it against success_criteria (JUDGE_MODEL)

Metrics (agent system only; simulator and judge tokens are reported separately as eval overhead):
  success rate  = scenarios the judge passed / scenarios run
  cost per task = (input tokens x input price + output tokens x output price) / scenarios, from pricing.json
  latency       = seconds per agent reply, reported as p50 and p95
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
logging.getLogger("google_adk").setLevel(logging.CRITICAL)

MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"
SIM_MODEL = os.getenv("SIM_MODEL") or MODEL
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or MODEL
DONE = "DONE"


# ---------- the three roles ----------

def simulate_user(client, scenario: dict, transcript: list[dict]) -> tuple[str, int]:
    """Next message from the simulated customer, or DONE. Returns (message, tokens used)."""
    history = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in transcript) or "(no messages yet)"
    prompt = (
        f"You are role-playing a customer contacting Harbor Supply Co. support.\n"
        f"Who you are: {scenario['persona']}\nYour goal: {scenario['goal']}\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Write only your next message to the support agent, in character, one or two sentences. "
        f"Don't reveal facts the agent should look up. If your goal has been met, or the agent clearly "
        f"can't help, reply with exactly {DONE}."
    )
    r = client.models.generate_content(model=SIM_MODEL, contents=prompt)
    return (r.text or DONE).strip(), (r.usage_metadata.total_token_count or 0) if r.usage_metadata else 0


def judge(client, scenario: dict, transcript: list[dict]) -> tuple[dict, int]:
    """Grade one conversation. Returns ({"success": bool, "reason": str}, tokens used)."""
    history = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in transcript)
    prompt = (
        "You are grading a customer-support conversation against a success rule. Be strict: "
        "every fact in the rule must be stated correctly by the AGENT, and anything the rule calls a failure fails.\n\n"
        f"Success rule: {scenario['success_criteria']}\n\nConversation:\n{history}"
    )
    r = client.models.generate_content(
        model=JUDGE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={"type": "object", "properties": {
                "success": {"type": "boolean"}, "reason": {"type": "string"}},
                "required": ["success", "reason"]},
        ),
    )
    return json.loads(r.text), (r.usage_metadata.total_token_count or 0) if r.usage_metadata else 0


async def agent_reply(runner: Runner, session_id: str, text: str, max_calls: int) -> dict:
    """Send one user message to the agent system; collect its reply, tokens, and timing."""
    reply, tokens_in, tokens_out, error = [], 0, 0, None
    start = time.perf_counter()
    try:
        async for event in runner.run_async(
                user_id="sim", session_id=session_id, run_config=RunConfig(max_llm_calls=max_calls),
                new_message=types.Content(role="user", parts=[types.Part(text=text)])):
            if event.usage_metadata:
                tokens_in += event.usage_metadata.prompt_token_count or 0
                tokens_out += event.usage_metadata.candidates_token_count or 0
            for part in (event.content.parts if event.content and event.content.parts else []):
                if part.text and not getattr(part, "thought", False) and event.author != "user":
                    reply.append(part.text.strip())
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return {"text": " ".join(reply) or "(no reply)", "tokens_in": tokens_in, "tokens_out": tokens_out,
            "seconds": time.perf_counter() - start, "error": error}


# ---------- one scenario ----------

async def run_scenario(root_agent, client, scenario: dict, max_turns: int, max_calls: int) -> dict:
    runner = Runner(app_name="harbor_eval", agent=root_agent,
                    session_service=InMemorySessionService(), auto_create_session=True)
    transcript, turns, overhead = [], [], 0
    for _ in range(max_turns):
        message, used = simulate_user(client, scenario, transcript)
        overhead += used
        if message.strip().upper().startswith(DONE):
            break
        transcript.append({"role": "user", "text": message})
        r = await agent_reply(runner, scenario["id"], message, max_calls)
        turns.append(r)
        transcript.append({"role": "agent", "text": r["text"]})
        if r["error"]:
            transcript.append({"role": "system", "text": f"Agent crashed: {r['error']}"})
            break
    verdict, used = judge(client, scenario, transcript)
    overhead += used
    return {"id": scenario["id"], "success": bool(verdict["success"]), "reason": verdict["reason"],
            "turns": len(turns), "tokens_in": sum(t["tokens_in"] for t in turns),
            "tokens_out": sum(t["tokens_out"] for t in turns),
            "reply_seconds": [round(t["seconds"], 2) for t in turns],
            "errors": [t["error"] for t in turns if t["error"]],
            "eval_overhead_tokens": overhead, "transcript": transcript}


# ---------- metrics ----------

def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the value below which p% of observations fall."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(results: list[dict], pricing: dict) -> dict:
    n = len(results)
    tokens_in = sum(r["tokens_in"] for r in results)
    tokens_out = sum(r["tokens_out"] for r in results)
    cost = tokens_in / 1e6 * pricing["input_per_million"] + tokens_out / 1e6 * pricing["output_per_million"]
    latencies = [s for r in results for s in r["reply_seconds"]]
    return {
        "scenarios": n,
        "success_rate": round(sum(r["success"] for r in results) / n, 3) if n else 0,
        "cost_per_task_usd": round(cost / n, 6) if n else 0,
        "tokens_per_task": round((tokens_in + tokens_out) / n) if n else 0,
        "latency_p50_s": round(percentile(latencies, 50), 2),
        "latency_p95_s": round(percentile(latencies, 95), 2),
        "crashes": sum(bool(r["errors"]) for r in results),
        "eval_overhead_tokens": sum(r["eval_overhead_tokens"] for r in results),
    }


def load_system(name: str):
    if name == "single":
        from single_agent.agent import root_agent
    else:
        from team_agent.agent import root_agent
    return root_agent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["single", "team"], required=True)
    parser.add_argument("--limit", type=int, help="run only the first N scenarios")
    parser.add_argument("--only", help="run one scenario id")
    parser.add_argument("--max-turns", type=int, default=4, help="most user messages per conversation")
    parser.add_argument("--max-calls", type=int, default=12, help="model-call budget per agent reply")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between scenarios (free-tier rate limits)")
    args = parser.parse_args()

    scenarios = json.loads((HERE / "scenarios.json").read_text())
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]
    scenarios = scenarios[: args.limit] if args.limit else scenarios
    pricing = json.loads((HERE / "pricing.json").read_text())
    root_agent, client = load_system(args.system), genai.Client()

    results = []
    for s in scenarios:
        r = await run_scenario(root_agent, client, s, args.max_turns, args.max_calls)
        results.append(r)
        print(f"{'PASS' if r['success'] else 'FAIL'}  {s['id']:<20} turns={r['turns']} "
              f"tokens={r['tokens_in'] + r['tokens_out']}  {r['reason'][:90]}")
        await asyncio.sleep(args.pause)

    summary = summarize(results, pricing)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.system}.json").write_text(json.dumps(
        {"system": args.system, "model": MODEL, "summary": summary, "results": results}, indent=2))
    print(f"\n{args.system}: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    print(f"Saved evals/results/{args.system}.json")


if __name__ == "__main__":
    asyncio.run(main())

"""Run every scenario through the support agent and print a readable trace of each.

    python run_scenarios.py                  # all scenarios
    python run_scenarios.py --only s3-where-is-it
    python run_scenarios.py --max-calls 10   # step budget per scenario (default 12)

Every trace is also saved as JSON under traces/, so you can compare runs.
Once Stage 1 adds tracing.py, the same runs also appear in Langfuse.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from support_agent.agent import root_agent

try:
    import tracing  # Stage 1 creates this: sends runs to Langfuse
except ImportError:
    tracing = None

logging.getLogger("google_adk").setLevel(logging.CRITICAL)  # errors are shown in the trace instead
ROOT = Path(__file__).resolve().parent


def describe(event) -> list[dict]:
    """Turn one ADK event into simple trace steps."""
    steps = []
    for part in (event.content.parts if event.content and event.content.parts else []):
        if part.function_call:
            steps.append({"kind": "tool_call", "tool": part.function_call.name, "args": dict(part.function_call.args or {})})
        elif part.function_response:
            steps.append({"kind": "tool_result", "tool": part.function_response.name, "result": part.function_response.response})
        elif part.text and not getattr(part, "thought", False):
            steps.append({"kind": "text", "text": part.text})
    if event.error_code or event.error_message:
        steps.append({"kind": "error", "error": f"{event.error_code}: {event.error_message}"})
    return steps


async def run_one(scenario: dict, max_calls: int) -> dict:
    runner = Runner(app_name="harbor_support", agent=root_agent,
                    session_service=InMemorySessionService(), auto_create_session=True)
    trace, tokens, llm_calls = [], 0, 0
    start = time.perf_counter()
    message = types.Content(role="user", parts=[types.Part(text=scenario["message"])])
    try:
        async for event in runner.run_async(user_id="customer", session_id=scenario["id"], new_message=message,
                                             run_config=RunConfig(max_llm_calls=max_calls)):
            if event.usage_metadata:
                llm_calls += 1
                tokens += event.usage_metadata.total_token_count or 0
            for step in describe(event):
                step["t"] = round(time.perf_counter() - start, 2)
                trace.append(step)
    except Exception as e:  # a crash is part of the trace, not the end of the script
        trace.append({"kind": "crash", "error": f"{type(e).__name__}: {e}", "t": round(time.perf_counter() - start, 2)})
    final = next((s["text"] for s in reversed(trace) if s["kind"] == "text"), None)
    return {"id": scenario["id"], "message": scenario["message"], "final_answer": final,
            "llm_calls": llm_calls, "tokens": tokens, "seconds": round(time.perf_counter() - start, 2), "trace": trace}


def print_trace(result: dict) -> None:
    print(f"\n=== {result['id']}: {result['message']}")
    for s in result["trace"]:
        if s["kind"] == "tool_call":
            print(f"  [{s['t']:>5}s] CALL   {s['tool']}({json.dumps(s['args'])})")
        elif s["kind"] == "tool_result":
            print(f"  [{s['t']:>5}s] RESULT {s['tool']} -> {json.dumps(s['result'], default=str)[:160]}")
        elif s["kind"] == "text":
            print(f"  [{s['t']:>5}s] SAYS   {s['text'].strip()[:300]}")
        else:
            print(f"  [{s['t']:>5}s] {s['kind'].upper():6} {s['error'][:200]}")
    print(f"  -- {result['llm_calls']} model calls, {result['tokens']} tokens, {result['seconds']}s")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run one scenario id")
    parser.add_argument("--max-calls", type=int, default=12, help="model-call budget per scenario")
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between scenarios (free-tier rate limits)")
    args = parser.parse_args()

    scenarios = json.loads((ROOT / "scenarios.json").read_text())
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]
    (ROOT / "traces").mkdir(exist_ok=True)
    for scenario in scenarios:
        result = await run_one(scenario, args.max_calls)
        print_trace(result)
        (ROOT / "traces" / f"{scenario['id']}.json").write_text(json.dumps(result, indent=2, default=str))
        await asyncio.sleep(args.pause)
    if tracing:
        tracing.flush()


if __name__ == "__main__":
    asyncio.run(main())

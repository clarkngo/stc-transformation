"""Plan-and-execute: write the whole plan first, then carry it out step by step.

This is the Stage 4 (by hand) piece. It makes *planning* an explicit,
inspectable artifact instead of something hidden inside each ReAct step.
It reuses the hand-built loop for each step, so no new loop code is needed.
"""

import json
import sys
import time

from google.genai import types

from . import react_loop
from .common import MODEL, count_tokens, get_client

PLANNER = (
    "Break the user's question into 2-4 short research steps. Each step should be "
    "answerable with a Wikipedia lookup or one calculation. Return only the steps."
)


def make_plan(question: str) -> tuple[list[str], int]:
    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER,
            response_mime_type="application/json",
            response_schema=list[str],
        ),
    )
    return json.loads(response.text), count_tokens(response)


def run(question: str, verbose: bool = False) -> dict:
    start = time.perf_counter()
    plan, tokens = make_plan(question)
    if verbose:
        print("PLAN:", *[f"  {i}. {s}" for i, s in enumerate(plan, 1)], sep="\n")

    findings, steps, tool_calls, log = [], 1, 0, [{"step": 1, "type": "plan", "plan": plan}]
    for i, step in enumerate(plan, 1):
        context = "\n".join(findings) or "(none yet)"
        sub = react_loop.run(
            f"Overall question: {question}\nFindings so far:\n{context}\n\nDo only this step: {step}",
            max_steps=4,
        )
        findings.append(f"Step {i} ({step}): {sub['answer']}")
        steps += sub["steps"]
        tool_calls += sub["tool_calls"]
        tokens += sub["tokens"]
        log.append({"step": i + 1, "type": "executed", "plan_step": step, "result": sub["answer"]})

    # Final synthesis from the findings only.
    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Question: {question}\nFindings:\n" + "\n".join(findings)
        + "\n\nGive a short final answer based only on these findings.",
    )
    tokens += count_tokens(response)
    log.append({"step": steps + 1, "type": "final", "text": response.text})
    return {"answer": response.text, "steps": steps + 1, "tool_calls": tool_calls, "tokens": tokens,
            "seconds": round(time.perf_counter() - start, 2), "log": log, "plan": plan}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many years before the first iPhone was the Space Needle built?"
    print(run(q, verbose=True)["answer"])

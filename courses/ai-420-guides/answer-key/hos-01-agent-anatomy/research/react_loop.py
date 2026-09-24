"""A ReAct agent loop written by hand, with no agent framework.

The four agent components are all visible here:
  - Reasoning: each generate_content call, where the model decides what to do next.
  - Planning:  implicit in ReAct (one step at a time). See plan_execute.py for explicit planning.
  - Tool use:  the function calls the model asks for, which *this code* executes.
  - Memory:    the `contents` list, which grows with every step and is resent each call.
"""

import sys
import time

from google.genai import types

from .common import MODEL, SYSTEM, count_tokens, get_client
from .tools import TOOLS, TOOLS_BY_NAME


def run(question: str, max_steps: int = 8, verbose: bool = False) -> dict:
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=TOOLS,
        # We run the loop ourselves, so turn off the SDK's automatic tool execution.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]  # memory
    log, tokens, tool_calls = [], 0, 0
    start = time.perf_counter()

    for step in range(1, max_steps + 1):
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)  # reasoning
        tokens += count_tokens(response)
        calls = response.function_calls or []

        if not calls:  # no tool requested: the model is done
            log.append({"step": step, "type": "final", "text": response.text})
            return _result(response.text, step, tool_calls, tokens, start, log, verbose)

        # Keep the model's own turn (including its function calls) in memory.
        contents.append(response.candidates[0].content)
        results = []
        for call in calls:  # tool use
            tool_calls += 1
            fn = TOOLS_BY_NAME.get(call.name)
            output = fn(**(call.args or {})) if fn else {"status": "error", "error": f"Unknown tool {call.name}"}
            log.append({"step": step, "type": "tool", "tool": call.name, "args": dict(call.args or {}), "result": output})
            results.append(types.Part.from_function_response(name=call.name, response=output))
        contents.append(types.Content(role="user", parts=results))  # observation goes back into memory

    # Step budget exhausted: say so honestly rather than looping forever.
    answer = f"I couldn't finish within {max_steps} steps. Partial findings are in the step log."
    log.append({"step": max_steps, "type": "stopped", "text": answer})
    return _result(answer, max_steps, tool_calls, tokens, start, log, verbose)


def _result(answer, steps, tool_calls, tokens, start, log, verbose):
    if verbose:
        for entry in log:
            if entry["type"] == "tool":
                print(f"  step {entry['step']}: {entry['tool']}({entry['args']}) -> {str(entry['result'])[:120]}")
            else:
                print(f"  step {entry['step']}: {entry['type']}")
    return {"answer": answer, "steps": steps, "tool_calls": tool_calls, "tokens": tokens,
            "seconds": round(time.perf_counter() - start, 2), "log": log}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many years before the first iPhone was the Space Needle built?"
    print(run(q, verbose=True)["answer"])

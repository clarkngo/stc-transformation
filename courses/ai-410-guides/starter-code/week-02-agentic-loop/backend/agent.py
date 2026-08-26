"""
The agentic loop: model decides -> tool runs -> result goes back to
the model -> repeat until the model returns a plain text answer.

This file is complete and working as-is for Week 2 — your job this
week is in tools.py, not here. Read this file to understand the
mechanism; you'll extend the loop itself in later weeks (Week 4 adds
retrieval, Week 6 adds guardrails).
"""

from anthropic import Anthropic

from tools import TOOLS, TOOL_FUNCTIONS

client = Anthropic()
MODEL = "claude-sonnet-5"
MAX_TURNS = 5  # hard ceiling so a misbehaving loop can't run forever


def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # Plain text answer — the model didn't need a tool.
            return "".join(block.text for block in response.content if block.type == "text")

        # The model wants to call one or more tools. Run each one and
        # send the results back before asking the model to continue.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            if fn is None:
                result = f"Error: no tool registered named '{block.name}'"
            else:
                result = fn(**block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return "I couldn't finish that within the allowed number of steps."

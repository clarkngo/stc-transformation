"""
The agentic loop: model decides -> tool runs -> result goes back to
the model -> repeat until the model returns a plain text answer.

This file is complete and working as-is for Week 2 — your job this
week is in tools.py, not here. Read this file to understand the
mechanism; you'll extend the loop itself in later weeks (Week 4 adds
retrieval, Week 6 adds guardrails).

Uses Gemini's Interactions API: each call returns an `interaction`
with a list of `steps`. A step of type "function_call" means the
model wants to run a tool; you run it locally and send a
"function_result" back via `previous_interaction_id` to continue the
same interaction.
"""

from dotenv import load_dotenv
from google import genai

from tools import TOOLS, TOOL_FUNCTIONS

load_dotenv()  # main.py also calls this, but agent.py is imported before
                # that runs — this module needs its own env vars loaded
                # before constructing the client below.
client = genai.Client()
MODEL = "gemini-flash-latest"
MAX_TURNS = 5  # hard ceiling so a misbehaving loop can't run forever


def run_agent(user_message: str) -> str:
    interaction = client.interactions.create(
        model=MODEL,
        input=user_message,
        tools=TOOLS,
    )

    for _ in range(MAX_TURNS):
        function_calls = [step for step in interaction.steps if step.type == "function_call"]

        if not function_calls:
            # Plain text answer — the model didn't need a tool.
            return interaction.output_text

        # The model wants to call one or more tools. Run each one and
        # send the results back before asking the model to continue.
        results = []
        for call in function_calls:
            fn = TOOL_FUNCTIONS.get(call.name)
            if fn is None:
                result = f"Error: no tool registered named '{call.name}'"
            else:
                result = fn(**call.arguments)
            results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": str(result)}],
                }
            )

        interaction = client.interactions.create(
            model=MODEL,
            input=results,
            tools=TOOLS,
            previous_interaction_id=interaction.id,
        )

    return "I couldn't finish that within the allowed number of steps."

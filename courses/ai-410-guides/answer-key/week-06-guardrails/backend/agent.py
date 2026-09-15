"""
The agentic loop — RAG (Week 4) is solved and wired in below. The
tool call is wrapped with the guardrail from guardrails.py using the
schema defined there.
"""

from dotenv import load_dotenv
from google import genai

from guardrails import ToolArgs, call_with_guardrail
from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

load_dotenv()  # main.py also calls this, but agent.py is imported before
                # that runs — this module needs its own env vars loaded
                # before constructing the client below.
client = genai.Client()
MODEL = "gemini-flash-latest"
MAX_TURNS = 5


def run_agent(user_message: str) -> str:
    chunks = retrieve(user_message, k=5)
    context = "\n\n".join(chunks)
    system_instruction = (
        "Use the following context to answer questions about Northwind "
        "Outfitters' policies and products. If a policy/product question "
        "isn't answered by the context, say you don't know rather than "
        "guessing. This restriction does not apply to your tools (e.g. "
        "calculate) — use them normally whenever they help, regardless "
        "of what's in the context below.\n\n"
        f"{context}"
    )

    interaction = client.interactions.create(
        model=MODEL,
        input=user_message,
        tools=TOOLS,
        system_instruction=system_instruction,
    )

    for _ in range(MAX_TURNS):
        function_calls = [step for step in interaction.steps if step.type == "function_call"]

        if not function_calls:
            return interaction.output_text

        results = []
        for call in function_calls:
            fn = TOOL_FUNCTIONS.get(call.name)

            if fn is None:
                result = f"Error: no tool registered named '{call.name}'"
            elif call.name == "calculate":
                validated = call_with_guardrail(lambda: call.arguments, ToolArgs)
                result = (
                    fn(**validated.model_dump())
                    if isinstance(validated, ToolArgs)
                    else validated  # the {"error": ...} dict from the guardrail
                )
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
            system_instruction=system_instruction,
        )

    return "I couldn't finish that within the allowed number of steps."

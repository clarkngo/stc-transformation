"""
The agentic loop — RAG (Week 4) is solved and wired in below. Your
Week 6 job: apply the guardrail from guardrails.py around the tool
call, using the schema you defined there.
"""

from dotenv import load_dotenv
from google import genai

from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

# TODO(week6): import your schema and call_with_guardrail once defined
# from guardrails import call_with_guardrail, ToolArgs

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
        "Use the following context to answer the user's question. "
        "If the answer isn't in the context, say you don't know.\n\n"
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
            else:
                # TODO(week6): replace this direct call with a guarded one, e.g.
                #
                #   validated = call_with_guardrail(lambda: call.arguments, ToolArgs)
                #   result = fn(**validated.model_dump()) if isinstance(validated, ToolArgs) else validated
                #
                # so malformed tool arguments get caught and retried instead
                # of crashing the loop or silently producing garbage.
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

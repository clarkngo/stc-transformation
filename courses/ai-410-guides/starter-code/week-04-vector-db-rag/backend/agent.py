"""
The agentic loop, extended with a RAG step. Everything from Week 2 is
unchanged except the TODO block below, where you'll inject retrieved
context before the first call to Gemini.
"""

from google import genai

from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

client = genai.Client()
MODEL = "gemini-flash-latest"
MAX_TURNS = 5


def run_agent(user_message: str) -> str:
    # TODO(week4): call retrieve(user_message) here and turn the
    # results into a system instruction so the model can ground its
    # answer in your documents. For example:
    #
    #   chunks = retrieve(user_message, k=5)
    #   context = "\n\n".join(chunks)
    #   system_instruction = (
    #       "Use the following context to answer the user's question. "
    #       "If the answer isn't in the context, say you don't know.\n\n"
    #       f"{context}"
    #   )
    #
    # Then pass system_instruction=system_instruction in BOTH
    # interactions.create() calls below (the initial one and the one
    # inside the loop) so the model keeps the context on every turn.

    interaction = client.interactions.create(
        model=MODEL,
        input=user_message,
        tools=TOOLS,
        # system_instruction=system_instruction,  # TODO(week4): uncomment once you build this above
    )

    for _ in range(MAX_TURNS):
        function_calls = [step for step in interaction.steps if step.type == "function_call"]

        if not function_calls:
            return interaction.output_text

        results = []
        for call in function_calls:
            fn = TOOL_FUNCTIONS.get(call.name)
            result = fn(**call.arguments) if fn else f"Error: no tool registered named '{call.name}'"
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
            # system_instruction=system_instruction,  # TODO(week4): uncomment here too
        )

    return "I couldn't finish that within the allowed number of steps."

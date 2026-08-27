"""
The agentic loop — RAG (Week 4) and the guardrail (Week 6) are both
solved and wired in below. Your Week 9 job: add tracing. See the
TODO comments for where to add @observe.
"""

from dotenv import load_dotenv
from google import genai

# TODO(week9): uncomment once you've set your LANGFUSE_* env vars
# from langfuse.decorators import observe

from guardrails import ToolArgs, call_with_guardrail
from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

load_dotenv()  # main.py also calls this, but agent.py is imported before
                # that runs — this module needs its own env vars loaded
                # before constructing the client below.
client = genai.Client()
MODEL = "gemini-flash-latest"
MAX_TURNS = 5


# TODO(week9): add @observe() above this function so each call becomes
# a trace in the Langfuse dashboard.
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
                validated = call_with_guardrail(lambda: call.arguments, ToolArgs)
                result = (
                    fn(**validated.model_dump())
                    if isinstance(validated, ToolArgs)
                    else validated  # the {"error": ...} dict from the guardrail
                )

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


# TODO(week9): add a simple RAG evaluation, e.g. a heuristic or a
# second cheap LLM call that scores whether `answer` actually used
# `context`, versus ignoring it. Call this from run_agent() and log
# the score (Langfuse can attach scores to a trace via
# langfuse_context.score_current_observation(...)).
#
# def score_faithfulness(answer: str, context: str) -> float:
#     ...

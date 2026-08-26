"""
The full agentic loop — RAG (Week 4), guardrails (Week 6), and
tracing (Week 9) are all solved and wired in below. Nothing new to
build here this week; Week 10 is about deployment, not new features.
"""

from google import genai
from langfuse.decorators import observe

from guardrails import ToolArgs, call_with_guardrail
from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

client = genai.Client()
MODEL = "gemini-flash-latest"
MAX_TURNS = 5


@observe()
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
                    else validated
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

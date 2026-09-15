"""
The agentic loop — RAG (Week 4) and the guardrail (Week 6) are both
solved and wired in below. Traced end to end, with a faithfulness
score logged on every call.
"""

import re

from dotenv import load_dotenv
from google import genai
from langfuse.decorators import langfuse_context, observe

from guardrails import ToolArgs, call_with_guardrail
from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

load_dotenv()  # main.py also calls this, but agent.py is imported before
                # that runs — this module needs its own env vars loaded
                # before constructing the client below.
client = genai.Client()
MODEL = "gemini-flash-latest"
MAX_TURNS = 5

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "at", "your", "you", "it", "this", "that", "be",
    "as", "with", "by", "from", "i", "don't", "know",
}


def score_faithfulness(answer: str, context: str) -> float:
    """
    A simple heuristic RAG evaluation. Scores what fraction of the
    "significant" words in the answer (lowercased, punctuation
    stripped, stopwords removed) also appear somewhere in the
    retrieved context. Not a substitute for a real judge model, but
    cheap, deterministic, and enough to flag an answer that's
    drifting away from what was actually retrieved.
    """
    answer_words = {w for w in re.findall(r"[a-z0-9']+", answer.lower()) if w not in _STOPWORDS}
    if not answer_words:
        return 1.0

    context_words = set(re.findall(r"[a-z0-9']+", context.lower()))
    grounded = answer_words & context_words
    return round(len(grounded) / len(answer_words), 3)


@observe()
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
            answer = interaction.output_text
            score = score_faithfulness(answer, context)
            try:
                langfuse_context.score_current_observation(name="faithfulness", value=score)
            except Exception:
                pass  # no-op if there's no active Langfuse trace (e.g. keys not configured)
            return answer

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
                    else validated
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

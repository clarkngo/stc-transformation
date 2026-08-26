"""
The agentic loop — RAG (Week 4) and the guardrail (Week 6) are both
solved and wired in below. Your Week 9 job: add tracing. See the
TODO comments for where to add @observe.
"""

from anthropic import Anthropic

# TODO(week9): uncomment once you've set your LANGFUSE_* env vars
# from langfuse.decorators import observe

from guardrails import ToolArgs, call_with_guardrail
from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

client = Anthropic()
MODEL = "claude-sonnet-5"
MAX_TURNS = 5


# TODO(week9): add @observe() above this function so each call becomes
# a trace in the Langfuse dashboard.
def run_agent(user_message: str) -> str:
    chunks = retrieve(user_message, k=5)
    context = "\n\n".join(chunks)
    system_prompt = (
        "Use the following context to answer the user's question. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"{context}"
    )

    messages = [{"role": "user", "content": user_message}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            system=system_prompt,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            if fn is None:
                result = f"Error: no tool registered named '{block.name}'"
            else:
                validated = call_with_guardrail(lambda: block.input, ToolArgs)
                result = (
                    fn(**validated.model_dump())
                    if isinstance(validated, ToolArgs)
                    else validated  # the {"error": ...} dict from the guardrail
                )

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
            )

        messages.append({"role": "user", "content": tool_results})

    return "I couldn't finish that within the allowed number of steps."


# TODO(week9): add a simple RAG evaluation, e.g. a heuristic or a
# second cheap LLM call that scores whether `answer` actually used
# `context`, versus ignoring it. Call this from run_agent() and log
# the score (Langfuse can attach scores to a trace via
# langfuse_context.score_current_observation(...)).
#
# def score_faithfulness(answer: str, context: str) -> float:
#     ...

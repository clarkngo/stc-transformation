"""
The agentic loop, extended with a RAG step. Everything from Week 2 is
unchanged except the TODO block below, where you'll inject retrieved
context before the first call to Claude.
"""

from anthropic import Anthropic

from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

client = Anthropic()
MODEL = "claude-sonnet-5"
MAX_TURNS = 5


def run_agent(user_message: str) -> str:
    # TODO(week4): call retrieve(user_message) here and turn the
    # results into a system prompt (or a prepended context message)
    # so the model can ground its answer in your documents. For example:
    #
    #   chunks = retrieve(user_message, k=5)
    #   context = "\n\n".join(chunks)
    #   system_prompt = (
    #       "Use the following context to answer the user's question. "
    #       "If the answer isn't in the context, say you don't know.\n\n"
    #       f"{context}"
    #   )
    #
    # Then pass system=system_prompt in the messages.create() call below.

    messages = [{"role": "user", "content": user_message}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
            # system=system_prompt,  # TODO(week4): uncomment once you build this above
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            result = fn(**block.input) if fn else f"Error: no tool registered named '{block.name}'"
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
            )

        messages.append({"role": "user", "content": tool_results})

    return "I couldn't finish that within the allowed number of steps."

"""
The agentic loop — RAG (Week 4) is solved and wired in below. Your
Week 6 job: apply the guardrail from guardrails.py around the tool
call, using the schema you defined there.
"""

from anthropic import Anthropic

from retrieval import retrieve
from tools import TOOLS, TOOL_FUNCTIONS

# TODO(week6): import your schema and call_with_guardrail once defined
# from guardrails import call_with_guardrail, ToolArgs

client = Anthropic()
MODEL = "claude-sonnet-5"
MAX_TURNS = 5


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

            # TODO(week6): replace this direct call with a guarded one, e.g.
            #
            #   validated = call_with_guardrail(lambda: block.input, ToolArgs)
            #   result = fn(**validated.model_dump()) if fn else "..."
            #
            # so malformed tool arguments get caught and retried instead
            # of crashing the loop or silently producing garbage.
            result = fn(**block.input) if fn else f"Error: no tool registered named '{block.name}'"

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
            )

        messages.append({"role": "user", "content": tool_results})

    return "I couldn't finish that within the allowed number of steps."

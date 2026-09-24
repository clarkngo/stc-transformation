"""Self-correction guards, attached to the agent as ADK callbacks (Stage 4, written by hand).

These are generic: they don't know about any specific bug. They turn the two
failure shapes seen in the traces into something the model can recover from.
"""

import json

MAX_REPEATS = 2


def error_as_observation(tool, args, tool_context, error):
    """on_tool_error_callback: a tool that raises becomes an error the model can read, not a crashed run."""
    return {
        "status": "error",
        "error": f"{tool.name} failed: {type(error).__name__}: {error}",
        "hint": "Check your arguments against the tool's description and try once more, or explain the problem to the customer.",
    }


def repeat_call_guard(tool, args, tool_context):
    """before_tool_callback: stop the agent from calling the same tool with the same arguments again and again.

    Returning a dict skips the real tool call and sends this dict back to the model instead.
    Returning None lets the call go ahead.
    """
    key = f"{tool.name}:{json.dumps(args, sort_keys=True, default=str)}"
    counts = dict(tool_context.state.get("temp:call_counts", {}))  # temp: = this invocation only
    counts[key] = counts.get(key, 0) + 1
    tool_context.state["temp:call_counts"] = counts
    if counts[key] > MAX_REPEATS:
        return {
            "status": "error",
            "error": f"You've already called {tool.name} with these exact arguments {MAX_REPEATS} times. "
                     "The result won't change. Stop calling it and answer with what you have.",
        }
    return None

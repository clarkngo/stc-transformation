"""
Tool definitions for the agentic loop.

Each tool has two parts:
  1. A schema (what Claude sees) — name, description, and input_schema.
     The description is what Claude reads to decide WHEN to call it,
     so be specific.
  2. A Python function (what actually runs) — registered in TOOL_FUNCTIONS
     under the same name as the schema.
"""


def _ping(**kwargs):
    """Dummy tool — proves the loop works before you add a real one."""
    return "pong"


# --- Schemas Claude sees -----------------------------------------------

TOOLS = [
    {
        "name": "ping",
        "description": "A no-op test tool that always returns 'pong'. Useful only for verifying the tool-calling loop is wired correctly.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    # TODO(week2): add your own tool here. For example, a calculator:
    #
    # {
    #     "name": "calculate",
    #     "description": "Evaluate a basic arithmetic expression, e.g. '342 * 87'.",
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "expression": {"type": "string"},
    #         },
    #         "required": ["expression"],
    #     },
    # },
]

# --- Functions that actually run ----------------------------------------

TOOL_FUNCTIONS = {
    "ping": _ping,
    # TODO(week2): register your function here, matching the schema name above.
    #
    # "calculate": _calculate,
}

# TODO(week2): implement the function for your new tool. Keep it small
# and deterministic — don't use a raw eval() on untrusted input beyond
# this exercise; a proper expression parser (e.g. the `ast` module or a
# small library) is the safe way to do this.
#
# def _calculate(expression: str, **kwargs):
#     ...

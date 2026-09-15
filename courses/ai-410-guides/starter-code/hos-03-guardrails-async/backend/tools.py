"""
Tool definitions for the agentic loop.

Each tool has two parts:
  1. A schema (what Gemini sees) — name, description, and a JSON-schema
     `parameters` object. The description is what the model reads to
     decide WHEN to call it, so be specific.
  2. A Python function (what actually runs) — registered in TOOL_FUNCTIONS
     under the same name as the schema.

`calculate` is the tool a Create/Scaffold pass with an AI assistant
would typically produce first. `word_count` is written by hand for
Understand & Refine — deliberately a different shape (a single string
argument, no arithmetic) so it can't just be copy-adapted from
`calculate` without actually understanding the registration pattern.
"""


def _calculate(expression: str, **kwargs):
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](eval_node(node.operand))
        raise ValueError(f"Unsupported expression: {expression!r}")

    try:
        return eval_node(ast.parse(expression, mode="eval").body)
    except Exception as e:
        return f"Couldn't evaluate {expression!r}: {e}"


def _word_count(text: str, **kwargs):
    return len(text.split())


# --- Schemas Gemini sees -----------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression, e.g. '342 * 87'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "word_count",
        "description": "Count the number of words in a piece of text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
]

# --- Functions that actually run ----------------------------------------

TOOL_FUNCTIONS = {
    "calculate": _calculate,
    "word_count": _word_count,
}

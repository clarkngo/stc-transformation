"""
Tool definitions for the agentic loop.

Each tool has two parts:
  1. A schema (what Gemini sees) — name, description, and a JSON-schema
     `parameters` object. The description is what the model reads to
     decide WHEN to call it, so be specific.
  2. A Python function (what actually runs) — registered in TOOL_FUNCTIONS
     under the same name as the schema.
"""

import ast
import operator


def _ping(**kwargs):
    """Dummy tool — proves the loop works before you add a real one."""
    return "pong"


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _calculate(expression: str, **kwargs):
    try:
        tree = ast.parse(expression, mode="eval")
        return _safe_eval(tree.body)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


# --- Schemas Gemini sees -----------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "ping",
        "description": "A no-op test tool that always returns 'pong'. Useful only for verifying the tool-calling loop is wired correctly.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression, e.g. '342 * 87'. Supports +, -, *, /.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    },
]

# --- Functions that actually run ----------------------------------------

TOOL_FUNCTIONS = {
    "ping": _ping,
    "calculate": _calculate,
}

"""
Tool definitions for the agentic loop.

Carried over from Week 2, already solved: a small "calculate" tool
alongside the original "ping" dummy tool. Nothing to do here this
week — your Week 4 work is in retrieval.py and agent.py.
"""

import ast
import operator


def _ping(**kwargs):
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


TOOLS = [
    {
        "name": "ping",
        "description": "A no-op test tool that always returns 'pong'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression, e.g. '342 * 87'. Supports +, -, *, /.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

TOOL_FUNCTIONS = {
    "ping": _ping,
    "calculate": _calculate,
}

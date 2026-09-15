"""
Validation/retry middleware — solved in Week 6, unchanged here.
"""

from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ToolArgs(BaseModel):
    """Example schema from Week 6 — adjust to match your own tool's args."""

    expression: str


def call_with_guardrail(
    fn: Callable[..., dict],
    schema: Type[T],
    *args,
    max_retries: int = 2,
    **kwargs,
) -> T | dict:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = fn(*args, **kwargs)
            return schema.model_validate(raw)
        except ValidationError as e:
            last_error = e
            print(f"[guardrail] validation failed on attempt {attempt + 1}: {e}")

    return {"error": "could not get a valid response", "last_error": str(last_error)}

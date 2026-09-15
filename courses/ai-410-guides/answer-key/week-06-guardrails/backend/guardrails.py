"""
Validation/retry middleware template. call_with_guardrail() is
complete and reusable as-is. ToolArgs is the schema for this week's
bottleneck: the model occasionally omits `expression` or sends a
non-string value when calling `calculate`.
"""

from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def call_with_guardrail(
    fn: Callable[..., dict],
    schema: Type[T],
    *args,
    max_retries: int = 2,
    correction_hint: Callable[[str], str] | None = None,
    **kwargs,
) -> T | dict:
    """
    Call `fn(*args, **kwargs)`, validate the result against `schema`,
    and retry (with a correction hint appended) if validation fails.

    Returns a validated instance of `schema` on success, or a plain
    dict with an "error" key if all retries are exhausted — never
    raises, so a caller can always handle the failure gracefully.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = fn(*args, **kwargs)
            return schema.model_validate(raw)
        except ValidationError as e:
            last_error = e
            print(f"[guardrail] validation failed on attempt {attempt + 1}: {e}")
            if correction_hint and attempt < max_retries:
                kwargs = correction_hint(str(e), *args, **kwargs) or kwargs

    return {"error": "could not get a valid response", "last_error": str(last_error)}


class ToolArgs(BaseModel):
    """Guards the `calculate` tool's arguments."""

    expression: str

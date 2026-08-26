"""
Validation/retry middleware template.

call_with_guardrail() is complete and reusable as-is — it's the
"template" the Week 6 guide refers to. Your job is to:
  1. Define a Pydantic schema for the output you want to validate
     (see the TODO below).
  2. Wrap the relevant call in agent.py with call_with_guardrail().
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
                # Give the caller a chance to feed the error back into
                # the next attempt (e.g. by adjusting a prompt).
                kwargs = correction_hint(str(e), *args, **kwargs) or kwargs

    return {"error": "could not get a valid response", "last_error": str(last_error)}


# TODO(week6): define a schema for the output you want to guard.
# Pick the bottleneck from your Week 5 review that's about bad/
# malformed model output — most students will have one about tool
# call arguments. Example:
#
# class ToolArgs(BaseModel):
#     city: str
#     units: str = "metric"
#
# Then in agent.py, wrap the relevant call:
#
#     result = call_with_guardrail(run_tool, ToolArgs, tool_input)

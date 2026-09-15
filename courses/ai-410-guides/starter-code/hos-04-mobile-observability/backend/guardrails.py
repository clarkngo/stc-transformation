"""
Validation/retry middleware for nondeterministic tool-call arguments,
plus the schema for this HOS's guarded tool.
"""

from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError, field_validator

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


class CalculateArgs(BaseModel):
    """Guards the `calculate` tool's arguments — the model sometimes
    omits `expression` entirely or sends a non-string value."""

    expression: str


class WordCountArgs(BaseModel):
    """Added by hand for Understand & Refine — extends the guardrail
    pattern to a second tool. Rejects an empty/whitespace-only string,
    which the tool itself would happily "count" as zero words instead
    of flagging as a bad call."""

    text: str

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v

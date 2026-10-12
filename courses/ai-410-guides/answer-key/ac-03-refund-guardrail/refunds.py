"""
Reference solution for Application Challenge 3 (faculty only).

Routing here is ONE defensible answer: up to $75 goes straight through,
and so does nothing else. A second refund on an order that already had
one is queued whatever its size (repeat refunds are how small-limit
rules get gamed). Other strong answers exist, e.g. a percentage of the
order instead of a dollar figure, or queueing every wholesale account.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator  # noqa: F401

import review_queue
import store_data

AUTO_REFUND_LIMIT = 75.00

REASONS = ["damaged", "wrong_item", "late_delivery", "not_as_described"]


class RefundArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^PB-\d{4}$")
    amount: float = Field(gt=0)
    reason: str
    note: str | None = Field(default=None, max_length=200)

    @field_validator("amount")
    @classmethod
    def at_most_two_decimals(cls, v: float) -> float:
        if round(v, 2) != v:
            raise ValueError("amount can have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def known_reason(cls, v: str) -> str:
        if v not in REASONS:
            raise ValueError(f"reason must be one of {', '.join(REASONS)}")
        return v


def _validation_message(e: ValidationError) -> str:
    problems = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
    return "Invalid refund call. Fix these and call again: " + "; ".join(problems)


def issue_refund(raw_args: dict, customer_id: str) -> dict:
    try:
        args = RefundArgs(**raw_args)
    except ValidationError as e:
        return {"error": _validation_message(e)}
    except TypeError as e:
        return {"error": f"Invalid refund call: {e}"}

    order = store_data.ORDERS.get(args.order_id)
    if order is None or order["customer_id"] != customer_id:
        return {"error": f"No order {args.order_id} on your account."}
    age = (store_data.now() - datetime.fromisoformat(order["placed"])).days
    if age > store_data.REFUND_WINDOW_DAYS:
        return {"error": f"Order {args.order_id} is {age} days old; refunds are only possible within "
                         f"{store_data.REFUND_WINDOW_DAYS} days."}
    remaining = round(order["total"] - order["refunded"], 2)
    if args.amount > remaining:
        return {"error": f"Only ${remaining:.2f} of order {args.order_id} can still be refunded."}

    needs_person = (
        args.amount > store_data.HARD_LIMIT
        or args.amount > AUTO_REFUND_LIMIT
        or order["refunded"] > 0  # a repeat refund on the same order
    )
    if needs_person:
        job_id = review_queue.enqueue("refund", {"customer_id": customer_id, **args.model_dump()})
        return {"status": "queued", "job_id": job_id, "amount": args.amount}

    order["refunded"] = round(order["refunded"] + args.amount, 2)
    return {"status": "refunded", "order_id": args.order_id, "amount": args.amount}

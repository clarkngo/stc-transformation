"""
YOUR WORK GOES HERE — the guardrail for Pacific Bean's refund tool.

The support chatbot can call one tool:

    issue_refund(order_id, amount, reason, note?)

The model fills in those arguments, and in testing it got them wrong in
all kinds of ways (see recorded_calls.json). Your job: make sure a bad
call can never move money, and route every good call to the right place.

The app supplies the logged-in customer_id (like AC 1's patient_id).
The model never chooses it.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator  # noqa: F401

import review_queue
import store_data

# YOUR DECISION (see the challenge spec): refunds up to this amount go
# straight through; anything larger goes to the human review queue.
# It can't be more than store_data.HARD_LIMIT.
AUTO_REFUND_LIMIT = 0.00

REASONS = ["damaged", "wrong_item", "late_delivery", "not_as_described"]

# 1. A Pydantic model for the tool's arguments: RefundArgs.
#    Required rules (the tests check these):
#      - order_id: "PB-" followed by exactly 4 digits
#      - amount: a number greater than 0, with at most 2 decimal places
#        ("twenty dollars" must fail)
#      - reason: one of REASONS
#      - note: optional text, at most 200 characters
#      - any field not listed above (for example "approved": true) is rejected
#
# class RefundArgs(BaseModel):
#     ...


def issue_refund(raw_args: dict, customer_id: str) -> dict:
    """Run one tool call from the model. Never raises.

    1. Validate raw_args with RefundArgs. If it fails, return
       {"error": "..."} naming every bad field, so the model can fix the
       call and retry (the HOS 3 pattern).
    2. Check it against the store's data:
         - unknown order, or not this customer's -> error (same message for both)
         - placed more than REFUND_WINDOW_DAYS days ago -> error
         - more than total minus refunded so far -> error that says the remaining amount
    3. Route it:
         - over HARD_LIMIT -> never applied: enqueue it for review and return
           {"status": "queued", "job_id": ..., "amount": ...}
         - your rules (AUTO_REFUND_LIMIT, plus anything else you decide) pick
           "refunded" or "queued"
         - "refunded" adds the amount to the order's `refunded` and returns
           {"status": "refunded", "order_id": ..., "amount": ...}

    Errors and queued jobs never change the order. A $5 "damaged" refund on
    a recent order must still go straight through.
    """
    raise NotImplementedError

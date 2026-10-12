"""
Provided tests — your code must pass all of these. Run:  pytest

Don't edit this file. Your own tests go in test_my_rules.py.
"""

import copy

import pytest

import refunds
import review_queue
import store_data

GOOD = {"order_id": "PB-1001", "amount": 5, "reason": "damaged"}


def rejected_cleanly(raw, customer="C-201"):
    """Call issue_refund and confirm it returned an error without touching anything."""
    before = copy.deepcopy(store_data.ORDERS)
    result = refunds.issue_refund(raw, customer)
    assert "error" in result, f"expected an error, got {result}"
    assert store_data.ORDERS == before, "an error must not change any order"
    assert review_queue.JOBS == [], "an error must not queue anything"
    return result


# --- RefundArgs: the schema --------------------------------------------

def test_valid_args_pass_and_note_is_optional():
    refunds.RefundArgs(**GOOD)
    refunds.RefundArgs(**GOOD, note="Bag torn on arrival")


@pytest.mark.parametrize("field, value", [
    ("order_id", "1001"),
    ("order_id", "PB-12345"),
    ("amount", "twenty dollars"),
    ("amount", 0),
    ("amount", -18),
    ("amount", 12.833),
    ("reason", "broken"),
    ("note", "x" * 201),
])
def test_bad_values_are_rejected(field, value):
    with pytest.raises(refunds.ValidationError):
        refunds.RefundArgs(**{**GOOD, field: value})


def test_unknown_fields_are_rejected():
    with pytest.raises(refunds.ValidationError):
        refunds.RefundArgs(**GOOD, approved=True)


# --- issue_refund: bad calls never move money ---------------------------

def test_invalid_call_names_every_bad_field():
    result = rejected_cleanly({"order_id": "PB-1001", "amount": "twenty dollars", "reason": "broken"})
    assert "amount" in result["error"] and "reason" in result["error"]


def test_someone_elses_order_looks_the_same_as_no_order():
    theirs = rejected_cleanly({**GOOD, "order_id": "PB-1003"})     # belongs to C-202
    missing = rejected_cleanly({**GOOD, "order_id": "PB-9999"})
    # Same message apart from the order number, so the bot can't reveal
    # whether someone else has that order.
    assert theirs["error"].replace("PB-1003", "PB-####") == missing["error"].replace("PB-9999", "PB-####")


def test_old_order_is_outside_the_window():
    rejected_cleanly({"order_id": "PB-1002", "amount": 10, "reason": "damaged"})   # 46 days ago


def test_cannot_refund_more_than_what_is_left():
    result = rejected_cleanly({"order_id": "PB-1003", "amount": 129, "reason": "damaged"}, "C-202")
    assert "104" in result["error"]   # 129.00 total, 25.00 already refunded


# --- issue_refund: good calls go to the right place ---------------------

def test_small_damaged_refund_goes_straight_through():
    result = refunds.issue_refund(GOOD, "C-201")
    assert result == {"status": "refunded", "order_id": "PB-1001", "amount": 5}
    assert store_data.ORDERS["PB-1001"]["refunded"] == 5


def test_refunds_add_up():
    refunds.issue_refund(GOOD, "C-201")
    result = rejected_cleanly({"order_id": "PB-1001", "amount": 38.5, "reason": "damaged"})
    assert "33.5" in result["error"]


def test_over_the_hard_limit_is_queued_never_applied():
    result = refunds.issue_refund({"order_id": "PB-1005", "amount": 1240, "reason": "damaged"}, "C-203")
    assert result["status"] == "queued" and result["amount"] == 1240
    assert store_data.ORDERS["PB-1005"]["refunded"] == 0
    assert [j["job_id"] for j in review_queue.JOBS] == [result["job_id"]]
    assert review_queue.JOBS[0]["payload"]["order_id"] == "PB-1005"


def test_auto_limit_is_set_and_within_the_hard_limit():
    assert 0 < refunds.AUTO_REFUND_LIMIT <= store_data.HARD_LIMIT, "set AUTO_REFUND_LIMIT to your chosen limit"

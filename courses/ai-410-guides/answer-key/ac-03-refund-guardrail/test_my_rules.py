"""Reference tests for the answer key's routing rules."""

import refunds
import review_queue
import store_data


def test_exactly_at_the_limit_goes_through():
    result = refunds.issue_refund({"order_id": "PB-1006", "amount": 75, "reason": "damaged"}, "C-203")
    assert result["status"] == "refunded"


def test_one_cent_over_the_limit_is_queued():
    result = refunds.issue_refund({"order_id": "PB-1006", "amount": 75.01, "reason": "damaged"}, "C-203")
    assert result["status"] == "queued"
    assert store_data.ORDERS["PB-1006"]["refunded"] == 0


def test_second_small_refund_on_the_same_order_is_queued():
    first = refunds.issue_refund({"order_id": "PB-1001", "amount": 10, "reason": "damaged"}, "C-201")
    second = refunds.issue_refund({"order_id": "PB-1001", "amount": 10, "reason": "damaged"}, "C-201")
    assert first["status"] == "refunded" and second["status"] == "queued"
    assert len(review_queue.JOBS) == 1

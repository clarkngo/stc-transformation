"""Reference tests for the answer key's late-cancellation rule."""

import clinic_data
from tools import book_appointment, cancel_appointment


def test_late_cancel_is_refused_and_slot_stays_booked():
    result = cancel_appointment("P1002", "S02")  # 3 hours away
    assert "error" in result
    assert "front desk" in result["error"]
    assert clinic_data.SLOTS["S02"]["booked_by"] == "P1002"


def test_just_inside_the_window_is_still_late():
    book_appointment("P1003", "S06")  # 23.5 hours away
    assert "error" in cancel_appointment("P1003", "S06")
    assert clinic_data.SLOTS["S06"]["booked_by"] == "P1003"


def test_just_outside_the_window_cancels():
    book_appointment("P1003", "S05")  # 25 hours away
    assert cancel_appointment("P1003", "S05")["cancelled"] is True

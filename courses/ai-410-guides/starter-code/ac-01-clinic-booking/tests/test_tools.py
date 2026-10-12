"""
Provided tests — your tools must pass all of these. Run:  pytest

Don't edit this file. Your own tests go in test_my_rules.py.
"""

import copy

import clinic_data
from tools import TOOLS, TOOL_FUNCTIONS, book_appointment, cancel_appointment, find_open_slots


def unchanged(fn, *args):
    """Call fn and confirm it returned an error without touching the data."""
    before = copy.deepcopy(clinic_data.SLOTS)
    result = fn(*args)
    assert "error" in result, f"expected an error, got {result}"
    assert clinic_data.SLOTS == before, "an error must not change any data"
    return result


# --- find_open_slots ---------------------------------------------------

def test_open_slots_for_a_day_sorted_and_skips_booked_and_past():
    result = find_open_slots("2026-11-02")
    ids = [s["slot_id"] for s in result["slots"]]
    assert ids == ["S03", "S04"]  # S01 already started, S02 booked


def test_open_slot_fields():
    slot = find_open_slots("2026-11-02")["slots"][0]
    assert slot == {
        "slot_id": "S03",
        "provider_id": "dr-patel",
        "provider_name": clinic_data.PROVIDERS["dr-patel"],
        "start": "2026-11-02T14:00",
    }


def test_open_slots_filtered_by_provider():
    result = find_open_slots("2026-11-03", "np-okafor")
    assert [s["slot_id"] for s in result["slots"]] == ["S06"]  # S07 is booked


def test_no_open_slots_is_an_empty_list_not_an_error():
    assert find_open_slots("2026-11-09") == {"slots": []}


def test_bad_date_is_an_error():
    assert "error" in find_open_slots("next Tuesday")


def test_unknown_provider_is_an_error():
    assert "error" in find_open_slots("2026-11-02", "dr-who")


# --- book_appointment --------------------------------------------------

def test_book_open_slot():
    result = book_appointment("P1003", "S05")
    assert result["booked"] is True
    assert result["slot_id"] == "S05"
    assert result["start"] == "2026-11-03T10:00"
    assert clinic_data.SLOTS["S05"]["booked_by"] == "P1003"


def test_booked_slot_disappears_from_open_slots():
    book_appointment("P1003", "S03")
    ids = [s["slot_id"] for s in find_open_slots("2026-11-02")["slots"]]
    assert "S03" not in ids


def test_cannot_book_a_taken_slot():
    unchanged(book_appointment, "P1003", "S02")


def test_cannot_book_a_slot_that_already_started():
    unchanged(book_appointment, "P1003", "S01")


def test_cannot_book_unknown_slot_or_patient():
    unchanged(book_appointment, "P1003", "S99")
    unchanged(book_appointment, "P9999", "S03")


# --- cancel_appointment (the required part) ----------------------------

def test_cancel_own_appointment_more_than_24h_away():
    result = cancel_appointment("P1003", "S10")
    assert result["cancelled"] is True
    assert result["slot_id"] == "S10"
    assert clinic_data.SLOTS["S10"]["booked_by"] is None


def test_cannot_cancel_someone_elses_appointment():
    unchanged(cancel_appointment, "P1001", "S10")  # S10 belongs to P1003


def test_cannot_cancel_unknown_or_open_slot():
    unchanged(cancel_appointment, "P1001", "S99")
    unchanged(cancel_appointment, "P1001", "S03")


# --- schemas -----------------------------------------------------------

def test_every_tool_has_a_schema():
    names = {t["name"] for t in TOOLS}
    assert names == set(TOOL_FUNCTIONS), f"schemas for {set(TOOL_FUNCTIONS) - names} are missing"


def test_schemas_are_well_formed():
    for tool in TOOLS:
        assert tool["type"] == "function"
        assert len(tool["description"]) >= 20, f"{tool['name']}: write a real description"
        params = tool["parameters"]
        assert params["type"] == "object"
        assert set(params.get("required", [])) <= set(params["properties"])


def test_model_cannot_choose_the_patient():
    for tool in TOOLS:
        assert "patient_id" not in tool["parameters"]["properties"], (
            f"{tool['name']}: the app supplies patient_id; the model must not"
        )

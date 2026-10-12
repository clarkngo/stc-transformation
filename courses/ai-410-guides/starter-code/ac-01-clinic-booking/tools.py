"""
YOUR WORK GOES HERE — the tools for the Cascade Family Clinic agent.

Each tool has two parts (same pattern as HOS 1):
  1. A Python function that actually runs.
  2. A schema in TOOLS — the only thing Gemini sees.

Rules for every tool (the tests check these):
  - Return a dict. On success, return the fields described below.
  - On failure, return {"error": "<message the model can read>"} —
    never raise, and never change any data.
  - Use clinic_data.now() for the current time, not datetime.now().
  - The app supplies the logged-in patient's ID (see agent.py). Do NOT
    put patient_id in any schema's parameters — the model must not be
    able to choose whose appointment it is acting on.
"""

import clinic_data


def find_open_slots(date: str, provider_id: str | None = None) -> dict:
    """Open slots on one day.

    date: "YYYY-MM-DD". provider_id: optional, a key of
    clinic_data.PROVIDERS.

    Success: {"slots": [{"slot_id", "provider_id", "provider_name",
    "start"}, ...]} — only slots that are not booked and haven't
    already started, sorted by start time. A day with no open slots
    returns {"slots": []}, not an error.

    Errors: a date that isn't YYYY-MM-DD; an unknown provider_id.
    """
    raise NotImplementedError


def book_appointment(patient_id: str, slot_id: str) -> dict:
    """Book an open slot for the logged-in patient.

    Success: {"booked": True, "slot_id", "provider_name", "start"},
    and the slot's booked_by becomes patient_id.

    Errors: unknown patient; unknown slot; slot already booked; slot
    already started.
    """
    raise NotImplementedError


def cancel_appointment(patient_id: str, slot_id: str) -> dict:
    """Cancel one of the logged-in patient's own appointments.

    Required:
      - Unknown slot, or a slot not booked by this patient -> error,
        nothing changes.
      - More than 24 hours away -> {"cancelled": True, "slot_id",
        "start"}, and the slot is open again.

    YOUR DECISION (see the challenge spec): what happens when the
    appointment is less than 24 hours away? Implement your rule here,
    test it in tests/test_my_rules.py, and defend it in DECISIONS.md.
    """
    raise NotImplementedError


# --- Schemas Gemini sees -----------------------------------------------
# Write one schema per tool, in the same format as HOS 1's TOOLS list:
#   {"type": "function", "name": ..., "description": ...,
#    "parameters": {"type": "object", "properties": {...}, "required": [...]}}
# The description is how the model decides WHEN to call the tool.

TOOLS = []

# --- Functions that actually run ----------------------------------------

TOOL_FUNCTIONS = {
    "find_open_slots": find_open_slots,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
}

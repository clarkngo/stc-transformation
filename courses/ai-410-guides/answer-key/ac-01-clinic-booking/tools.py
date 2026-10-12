"""
Reference solution for Application Challenge 1 (faculty only).

The late-cancellation rule here is ONE defensible answer, not the
answer: inside the 24-hour window the agent does not cancel. It returns
an error explaining the $40 fee and tells the patient to call the front
desk, so a person — not the model — decides whether to charge or waive
it. Other strong answers exist (cancel and flag the fee; ask the patient
to confirm the fee first via a confirm_late_fee argument).
"""

from datetime import datetime

import clinic_data


def _err(message: str) -> dict:
    return {"error": message}


def find_open_slots(date: str, provider_id: str | None = None) -> dict:
    try:
        day = datetime.strptime(date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return _err(f"Date must be YYYY-MM-DD, got {date!r}.")
    if provider_id is not None and provider_id not in clinic_data.PROVIDERS:
        return _err(f"Unknown provider {provider_id!r}. Known: {', '.join(clinic_data.PROVIDERS)}.")
    slots = []
    for slot_id, slot in clinic_data.SLOTS.items():
        start = clinic_data.slot_start(slot_id)
        if start.date() != day or slot["booked_by"] is not None or start < clinic_data.now():
            continue
        if provider_id is not None and slot["provider_id"] != provider_id:
            continue
        slots.append({
            "slot_id": slot_id,
            "provider_id": slot["provider_id"],
            "provider_name": clinic_data.PROVIDERS[slot["provider_id"]],
            "start": slot["start"],
        })
    return {"slots": sorted(slots, key=lambda s: s["start"])}


def book_appointment(patient_id: str, slot_id: str) -> dict:
    if patient_id not in clinic_data.PATIENTS:
        return _err("Unknown patient.")
    slot = clinic_data.SLOTS.get(slot_id)
    if slot is None:
        return _err(f"No slot {slot_id!r}. Use find_open_slots to get valid slot IDs.")
    if slot["booked_by"] is not None:
        return _err(f"Slot {slot_id} is already booked.")
    if clinic_data.slot_start(slot_id) < clinic_data.now():
        return _err(f"Slot {slot_id} has already started.")
    slot["booked_by"] = patient_id
    return {"booked": True, "slot_id": slot_id,
            "provider_name": clinic_data.PROVIDERS[slot["provider_id"]], "start": slot["start"]}


def cancel_appointment(patient_id: str, slot_id: str) -> dict:
    slot = clinic_data.SLOTS.get(slot_id)
    if slot is None or slot["booked_by"] != patient_id:
        # Same message for "doesn't exist" and "not yours": don't reveal
        # whether someone else holds the slot.
        return _err(f"You don't have an appointment {slot_id!r}.")
    hours_away = (clinic_data.slot_start(slot_id) - clinic_data.now()).total_seconds() / 3600
    if hours_away < clinic_data.LATE_CANCEL_WINDOW_HOURS:
        return _err(
            f"This appointment is less than {clinic_data.LATE_CANCEL_WINDOW_HOURS} hours away, "
            f"so cancelling it carries a ${clinic_data.LATE_CANCEL_FEE} fee. The chat can't apply "
            "or waive that fee — please call the front desk to cancel."
        )
    slot["booked_by"] = None
    return {"cancelled": True, "slot_id": slot_id, "start": slot["start"]}


TOOLS = [
    {
        "type": "function",
        "name": "find_open_slots",
        "description": "List open appointment slots at Cascade Family Clinic on one date, "
                       "optionally for one provider. Call this before booking to get valid slot IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The day, as YYYY-MM-DD."},
                "provider_id": {"type": "string", "enum": list(clinic_data.PROVIDERS),
                                "description": "Optional: only this provider's slots."},
            },
            "required": ["date"],
        },
    },
    {
        "type": "function",
        "name": "book_appointment",
        "description": "Book one open slot for the patient you are talking with. "
                       "Only call this after the patient has chosen a specific slot.",
        "parameters": {
            "type": "object",
            "properties": {"slot_id": {"type": "string", "description": "A slot_id from find_open_slots, e.g. 'S05'."}},
            "required": ["slot_id"],
        },
    },
    {
        "type": "function",
        "name": "cancel_appointment",
        "description": "Cancel one of the current patient's own appointments. Appointments "
                       "less than 24 hours away can't be cancelled here; the result will say so.",
        "parameters": {
            "type": "object",
            "properties": {"slot_id": {"type": "string", "description": "The slot_id of the appointment to cancel."}},
            "required": ["slot_id"],
        },
    },
]

TOOL_FUNCTIONS = {
    "find_open_slots": find_open_slots,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
}

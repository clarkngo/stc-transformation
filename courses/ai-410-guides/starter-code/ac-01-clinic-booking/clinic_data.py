"""
Cascade Family Clinic — the clinic's data, kept in memory.

You don't need to change this file. Read it so you know what your
tools are working with.

The clinic runs on a FIXED clock (see `now()`) so the tests give the
same answer every time. Your tools must use `clinic_data.now()`,
never `datetime.now()`.
"""

import copy
from datetime import datetime

CLINIC_NOW = datetime(2026, 11, 2, 9, 0)  # Monday, 9:00 a.m.

LATE_CANCEL_WINDOW_HOURS = 24
LATE_CANCEL_FEE = 40  # dollars

PROVIDERS = {
    "dr-patel": "Dr. Anika Patel (family medicine)",
    "np-okafor": "Chidi Okafor, NP (nurse practitioner)",
}

# Fictional patients. The notes field is the kind of thing a real
# record holds — think about who should be able to read it.
PATIENTS = {
    "P1001": {"name": "Maria Lopez", "phone": "206-555-0142", "dob": "1988-04-12",
              "notes": "Penicillin allergy. Follow-up on blood pressure medication."},
    "P1002": {"name": "James Chen", "phone": "206-555-0177", "dob": "1975-09-30",
              "notes": "Type 2 diabetes. A1C check due."},
    "P1003": {"name": "Aisha Mohammed", "phone": "425-555-0119", "dob": "2001-01-08",
              "notes": "Sports physical. No known allergies."},
}

# slot_id -> slot. `booked_by` is a patient ID, or None if the slot is open.
_INITIAL_SLOTS = {
    "S01": {"provider_id": "dr-patel",  "start": "2026-11-02T08:30", "booked_by": None},
    "S02": {"provider_id": "dr-patel",  "start": "2026-11-02T12:00", "booked_by": "P1002"},
    "S03": {"provider_id": "dr-patel",  "start": "2026-11-02T14:00", "booked_by": None},
    "S04": {"provider_id": "np-okafor", "start": "2026-11-02T15:30", "booked_by": None},
    "S05": {"provider_id": "dr-patel",  "start": "2026-11-03T10:00", "booked_by": None},
    "S06": {"provider_id": "np-okafor", "start": "2026-11-03T08:30", "booked_by": None},
    "S07": {"provider_id": "np-okafor", "start": "2026-11-03T13:00", "booked_by": "P1001"},
    "S08": {"provider_id": "dr-patel",  "start": "2026-11-04T09:00", "booked_by": None},
    "S09": {"provider_id": "np-okafor", "start": "2026-11-04T11:00", "booked_by": None},
    "S10": {"provider_id": "dr-patel",  "start": "2026-11-05T16:00", "booked_by": "P1003"},
}

SLOTS = copy.deepcopy(_INITIAL_SLOTS)


def now() -> datetime:
    """The clinic's current time. Use this instead of datetime.now()."""
    return CLINIC_NOW


def slot_start(slot_id: str) -> datetime:
    return datetime.fromisoformat(SLOTS[slot_id]["start"])


def reset() -> None:
    """Put every slot back the way it started. The tests call this."""
    SLOTS.clear()
    SLOTS.update(copy.deepcopy(_INITIAL_SLOTS))

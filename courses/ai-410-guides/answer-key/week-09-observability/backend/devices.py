"""
Minimal device/push-token store. Complete as-is — an in-memory dict
is fine for this course; a real app would use a database table.
Restarting the server clears registered devices.
"""

_DEVICE_TOKENS: dict[str, str] = {}


def register_push_token(device_id: str, push_token: str) -> None:
    _DEVICE_TOKENS[device_id] = push_token


def get_push_token(device_id: str) -> str | None:
    return _DEVICE_TOKENS.get(device_id)

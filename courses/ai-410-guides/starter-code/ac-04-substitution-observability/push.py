"""A stand-in for push notifications: messages are recorded, not sent."""

SENT: list[dict] = []


def send(customer_id: str, message: str) -> None:
    SENT.append({"customer_id": customer_id, "message": message})


def reset() -> None:
    SENT.clear()

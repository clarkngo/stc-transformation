"""
Sends a push notification via Expo's push API. Complete as-is.
"""

import requests

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def send_push_notification(push_token: str, title: str, body: str) -> None:
    response = requests.post(
        EXPO_PUSH_URL,
        json={"to": push_token, "title": title, "body": body},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()

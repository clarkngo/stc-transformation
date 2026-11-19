"""
Device/push-token store, kept in Redis.

The API registers tokens (POST /register-device), but the "document
ready" push is sent by the RQ worker, which runs in a separate process.
A plain dict would live only in the API's memory, so the worker would
always find it empty and never send a push. Redis is already running
for the job queue, and both processes can reach it. A real app might
use a database table instead.
"""

from queue_setup import redis_conn

_KEY = "ai410:push-tokens"  # one Redis hash: device_id -> push token


def register_push_token(device_id: str, push_token: str) -> None:
    redis_conn.hset(_KEY, device_id, push_token)


def get_push_token(device_id: str) -> str | None:
    token = redis_conn.hget(_KEY, device_id)
    return token.decode() if token else None

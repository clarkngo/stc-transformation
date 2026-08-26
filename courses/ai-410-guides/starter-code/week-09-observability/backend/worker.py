"""
Run this in its OWN terminal, separate from `uvicorn main:app`:

    python worker.py

This process pulls jobs off the Redis queue and runs them. It is not
started automatically by the API — that's the key new mental model
this week. If jobs sit "queued" forever, this almost always means you
forgot to start this process.
"""

from rq import Worker

from queue_setup import queue, redis_conn

if __name__ == "__main__":
    worker = Worker([queue], connection=redis_conn)
    worker.work()

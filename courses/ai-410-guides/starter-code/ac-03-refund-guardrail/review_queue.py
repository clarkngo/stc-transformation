"""
The human review queue — a stand-in for HOS 3's Redis queue.

Like rq, `enqueue` returns a job ID immediately and the work happens
later: a person at Pacific Bean reviews each job and approves or
denies it. Here the queue is just a list, so the tests run offline.

You don't need to change this file.
"""

JOBS: list[dict] = []


def enqueue(kind: str, payload: dict) -> str:
    job_id = f"job-{len(JOBS) + 1:03d}"
    JOBS.append({"job_id": job_id, "kind": kind, "payload": dict(payload), "status": "queued"})
    return job_id


def reset() -> None:
    JOBS.clear()

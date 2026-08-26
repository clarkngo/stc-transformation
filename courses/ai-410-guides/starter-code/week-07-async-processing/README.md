# Week 7 Starter — Asynchronous Data Processing

Guide: [`../../week-07-async-processing.html`](../../week-07-async-processing.html)

## What's already working

- Chat with RAG + guardrails (Weeks 4-6, solved)
- `backend/queue_setup.py` — Redis/RQ connection, complete
- `backend/worker.py` — a working worker entry point (run it in its own terminal)
- `backend/main.py` — a `/documents` endpoint that runs ingestion **synchronously** (the thing you're about to fix)

## What you'll build this week

1. `backend/jobs.py` — implement `ingest_document_job()` (chunk → embed → insert), following the `# TODO(week7)` comment.
2. `backend/main.py` — change `/documents` to `queue.enqueue(...)` instead of calling the job function directly, and implement the `/jobs/{job_id}` status endpoint.

## Run it

You need **two terminals** running at once:

```bash
# Terminal 1 — the API
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys + REDIS_URL
./venv/bin/uvicorn main:app --reload

# Terminal 2 — the worker (separate process!)
cd backend
./venv/bin/python worker.py
```

Upload a document, confirm you get a job ID back instantly, then poll `/jobs/{id}` until it's "finished" — while sending normal chat messages the whole time to prove the app stayed responsive.

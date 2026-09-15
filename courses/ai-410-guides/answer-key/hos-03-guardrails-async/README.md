# HOS 3 Answer Key — Guardrails & Asynchronous Processing

Guide: [`../../hos-03-guardrails-async.html`](../../hos-03-guardrails-async.html)
Starter: [`../../starter-code/hos-03-guardrails-async`](../../starter-code/hos-03-guardrails-async)

Reference solution, built on top of HOS 2's RAG + tool-calling app.

## What's here

- `backend/guardrails.py` — `call_with_guardrail()` (complete, reusable) plus `CalculateArgs` (guards the `calculate` tool, a typical Create/Scaffold result) and `WordCountArgs` (guards `word_count`, added **by hand** for Understand & Refine — rejects an empty/whitespace-only string).
- `backend/agent.py` — HOS 2's loop, with both tool calls routed through the guardrail.
- `backend/jobs.py`, `queue_setup.py`, `worker.py` — document ingestion moved to a background job on a Redis queue.
- `backend/main.py` — `/documents` now enqueues instead of blocking; `/jobs/{id}` polls status.
- `EVALUATE.md` / `ANALYZE.md` — reference notes for the two written stages.

## Run it

Needs a Redis instance — the free Upstash tier works, no local Redis install required. Set `REDIS_URL` in `.env`.

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY and REDIS_URL
./venv/bin/python ingest.py

# Terminal 1 — API
./venv/bin/uvicorn main:app --reload

# Terminal 2 — worker (separate process!)
./venv/bin/python worker.py
```

In a third terminal, the frontend:

```bash
cd frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

Upload a document via `POST /documents`, confirm you get a `job_id` back instantly, then poll `/jobs/{id}` until it's `"finished"` — while sending normal chat messages the whole time to prove the app stayed responsive.

**Testing locally on macOS (not Codespaces):** if a job goes straight to `"failed"` with a worker-log crash mentioning `NSMutableString` / `fork()`, that's a known macOS Objective-C runtime + RQ forking-worker interaction, not a bug in this code — run the worker with `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python worker.py`. This doesn't happen in Codespaces (Linux), which is the recommended environment anyway.

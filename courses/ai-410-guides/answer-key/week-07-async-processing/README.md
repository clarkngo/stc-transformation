# Week 7 Answer Key — Asynchronous Data Processing

Guide: [`../../week-07-async-processing.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/week-07-async-processing.html)
Starter: [`../../starter-code/week-07-async-processing`](../../starter-code/week-07-async-processing)

Reference solution for SME/instructor verification — not what students should be given. Built on top of Week 6's guardrailed RAG app.

## What's solved here (vs. the starter)

- `backend/jobs.py`'s `ingest_document_job()` — chunk, embed, insert, self-contained so the worker process can import it independently of the API process.
- `backend/main.py` — `POST /documents` now enqueues the job (`queue.enqueue(...)`) and returns a `job_id` immediately instead of running ingestion inline; `GET /jobs/{id}` reports status via `rq`'s `Job.fetch`.

## Run it

Needs a Redis instance — the free Upstash tier works, no local install required.

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY and REDIS_URL
./venv/bin/python ingest.py
```

Then two separate terminals (in Codespaces, a new tab always starts back at the repo root, so `cd backend` again in each one):

```bash
# Terminal 1 — API
cd backend
./venv/bin/uvicorn main:app --reload
```
```bash
# Terminal 2 — worker (separate process!)
cd backend
./venv/bin/python worker.py
```

Upload a document, confirm you get a job ID back instantly, then poll `/jobs/{id}` until it's "finished" — while sending normal chat messages the whole time to prove the app stayed responsive.

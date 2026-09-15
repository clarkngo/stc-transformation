# Week 9 Answer Key — Observability & Evaluation

Guide: [`../../week-09-observability.html`](../../week-09-observability.html)
Starter: [`../../starter-code/week-09-observability`](../../starter-code/week-09-observability)

Reference solution for SME/instructor verification — not what students should be given. Built on top of Week 8's mobile/push app.

## What's solved here (vs. the starter)

- `backend/agent.py`, `retrieval.py` — decorated with Langfuse's `@observe()`, so `run_agent` and `retrieve` each produce their own span, nested under one trace per request.
- `backend/agent.py`'s `score_faithfulness(answer, context)` — a lexical-overlap heuristic scoring how much of an answer is grounded in the retrieved context, logged against the current trace via `langfuse_context.score_current_observation(...)`.

## Run it

Needs Redis (Week 7) plus a free [Langfuse](https://cloud.langfuse.com) project. The app works without Langfuse keys set — tracing disables itself with a log line — but you won't see anything in a dashboard until they're configured.

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY, REDIS_URL, LANGFUSE_*
./venv/bin/python ingest.py
```

Then two separate terminals (in Codespaces, a new tab always starts back at the repo root, so `cd backend` again in each one):

```bash
# Terminal 1 — API
cd backend
./venv/bin/uvicorn main:app --reload
```
```bash
# Terminal 2 — worker
cd backend
./venv/bin/python worker.py
```

Send a few chat messages, then check your Langfuse dashboard's Traces tab — each request should show retrieval and generation as separate nested spans, with a faithfulness score attached. Ask something the docs don't cover and see whether the score reflects it.

# Week 9 Starter — Observability & Evaluation

Guide: [`../../week-09-observability.html`](../../week-09-observability.html)

## What's already working

- The full pipeline through Week 8 (RAG, guardrails, async ingestion, push notifications)
- `langfuse` is already in `backend/requirements.txt` — just needs your keys in `.env`

## What you'll build this week

1. `backend/agent.py` — follow the `# TODO(week9)` comments: import `observe` and decorate `run_agent()`.
2. `backend/retrieval.py` — decorate `retrieve()` the same way, so it shows up as its own step in the trace.
3. `backend/agent.py` — add a simple `score_faithfulness()` check (heuristic or a second cheap LLM call) and log it.
4. **Both the API and `worker.py` need the Langfuse env vars** — they're separate processes.

## Run it

Same as Week 8 — **GitHub Codespaces recommended** — plus your Langfuse keys added to `backend/.env`. Send a few chat requests, then open your Langfuse dashboard and find the traces. Deliberately ask something outside your document set, find that trace, and write a short critique of what it shows.

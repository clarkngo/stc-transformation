# HOS 3 — Guardrails & Asynchronous Processing

Guide: [`../../hos-03-guardrails-async.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/hos-03-guardrails-async.html)

## What's already working

Your HOS 2 result, carried forward: RAG-grounded chat with tool calling (`backend/agent.py`, `retrieval.py`, `ingest.py`, `tools.py`, `main.py`, `frontend/`). Nothing to change here unless you want to.

## What's new (and blank on purpose)

No `guardrails.py`, no `jobs.py`, no `queue_setup.py`, no `worker.py` — **you build those with an AI assistant.** A free Redis instance ([Upstash](https://upstash.com), no credit card) is needed for the async half; see `backend/.env.example`.

## What you'll build

1. A validation/retry guardrail around one tool's arguments, so a malformed model output gets caught and retried instead of crashing.
2. Moving document ingestion to a background job (Redis/RQ), with a status endpoint instead of a blocking request.

Follow the guide's four stages: **Create/Scaffold → Evaluate → Analyze → Understand & Refine.**

# Week 6 Starter — Probabilistic Logic & Guardrails

Guide: [`../../week-06-guardrails.html`](../../week-06-guardrails.html)

## What's already working

- RAG is fully wired (Week 4's `retrieval.py` and `agent.py` are solved)
- `backend/guardrails.py` — a complete, reusable `call_with_guardrail()` retry/validate wrapper

## What you'll build this week

1. In `backend/guardrails.py`, define a Pydantic schema for the output you want to guard — pick the malformed-output bottleneck from your Week 5 review.
2. In `backend/agent.py`, follow the `# TODO(week6)` comment to wrap the tool call with `call_with_guardrail()`.

## Run it

Same setup as Week 4 (run `schema.sql`, then `./venv/bin/python ingest.py`, then `./venv/bin/uvicorn main:app --reload`).

Test it on purpose: feed the agent a prompt likely to produce malformed tool arguments and confirm the guardrail catches it — either self-correcting on retry, or failing with a clear message instead of crashing.

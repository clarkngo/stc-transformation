# HOS 4 — Mobile Client, Push & Observability

Guide: [`../../hos-04-mobile-observability.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/hos-04-mobile-observability.html)

## What's already working

Your HOS 3 result, carried forward: RAG-grounded, guardrailed chat with async document ingestion (`backend/agent.py`, `guardrails.py`, `jobs.py`, `queue_setup.py`, `worker.py`, `main.py`, `retrieval.py`, `ingest.py`, `tools.py`, `frontend/`). Nothing to change here unless you want to.

## What's new (and blank on purpose)

No `mobile/` app, no device registration, no tracing — **you build those with an AI assistant.** A free [Langfuse](https://cloud.langfuse.com) project (no credit card) is needed for observability; see `backend/.env.example`.

**Heads up before you scaffold the mobile client:** requesting a push token (`getExpoPushTokenAsync()`) requires an EAS project ID as of Expo SDK 49+, and a fresh project doesn't have one until you run `npx eas init` once inside `mobile/` (free Expo account, no credit card — see the [Account & Service Setup Guide](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/account-setup-guide.html#expo)). If your AI-scaffolded app crashes the moment it asks for a push token, this is almost certainly why — the guide's Common Snags section covers it.

## What you'll build

1. An Expo (React Native) mobile client wired to your backend, that requests push-notification permission and registers the device.
2. A `POST /register-device` endpoint storing the token, and a push notification sent when a document-ingestion job finishes.
3. Request tracing on the agent and retrieval functions, plus a hand-written faithfulness check logged against every trace.

Follow the guide's four stages: **Create/Scaffold → Evaluate → Analyze → Understand & Refine.**

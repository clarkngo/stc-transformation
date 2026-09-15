# HOS 2 — Embeddings, Chunking & RAG Pipelines

Guide: [`../../hos-02-embeddings-rag.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/hos-02-embeddings-rag.html)

## What's already working

Your HOS 1 result, carried forward: `backend/main.py`, `backend/agent.py`, `backend/tools.py` (a chat app with an agentic tool-calling loop), and `frontend/app.py`. Nothing to change here unless you want to.

## What's new (and blank on purpose)

`backend/sample_docs/` — five sample documents for a fictional outdoor-gear company. No `ingest.py`, no `retrieval.py`, no vector database wiring — **you build those with an AI assistant**, same as HOS 1's Create/Scaffold stage.

## What you'll build

Chunk and embed the sample docs into a local vector database, then wire retrieval into the existing agent so it grounds its answers in them. Follow the guide's four stages: **Create/Scaffold → Evaluate → Analyze → Understand & Refine.**

Get a free Gemini API key (no credit card required) at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) if you don't already have one from HOS 1.

# Week 3 Starter — Embeddings & Chunking

Guide: [`../../week-03-embeddings-chunking.html`](../../week-03-embeddings-chunking.html)

This week is a **standalone sandbox** — it isn't wired into the chat app. You're building intuition for chunking and embeddings in isolation before connecting anything to the live app in Week 4.

## What's already here

- `sandbox/sample_docs/` — five short sample documents to chunk and embed
- `sandbox/ingest_sandbox.py` — a script with the chunking and embedding functions stubbed out, plus a working similarity-check harness at the bottom

## What you'll build this week

Follow the `# TODO(week3)` comments in `ingest_sandbox.py`:
1. `chunk_text()` — split a document into ~300–500 token pieces with a little overlap
2. `embed_chunks()` — call the Voyage AI embeddings API on your chunks

## Run it

```bash
cd sandbox
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your VOYAGE_API_KEY
python ingest_sandbox.py
```

It will print a ranked list of chunks by similarity to a sample query — check that the top result is genuinely the most relevant one.

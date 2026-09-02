# Week 4 Starter — Vector Databases & RAG Pipelines

Guide: [`../../week-04-vector-db-rag.html`](../../week-04-vector-db-rag.html)

## What's already working

- Weeks 1-2's agent loop, with a working `calculate` tool (`backend/tools.py`)
- `backend/schema.sql` — the Supabase table schema (run this in the Supabase SQL editor first)
- `backend/ingest.py` — chunks the sample docs in `backend/sample_docs/` and inserts them with embeddings into your Supabase `documents` table
- `backend/sample_docs/` — the same 5 sample documents from Week 3

## What you'll build this week

1. `backend/retrieval.py` — implement `retrieve()` (embed the query, run a similarity search, return the matching chunks). Follow the `# TODO(week4)` comment.
2. `backend/agent.py` — wire `retrieve()` into the loop so its results are injected as context before the first call to Gemini. Follow the `# TODO(week4)` comment.

## Run it

First, in the Supabase SQL editor, run `backend/schema.sql` — this is the same regardless of how you run the app.

**Recommended: GitHub Codespaces.** Push this folder to its own repo, then **Code → Codespaces → Create codespace on main** — `.devcontainer/devcontainer.json` installs `backend/requirements.txt` and creates `backend/.env` automatically. Add your `GEMINI_API_KEY` and Supabase `DATABASE_URL` to `backend/.env`, then:
```bash
cd backend
python ingest.py          # load the sample docs into your vector DB
uvicorn main:app --reload
```

**Running locally instead?**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys + DATABASE_URL
./venv/bin/python ingest.py
./venv/bin/uvicorn main:app --reload
```

Either way, ask something that can only be answered from the sample docs (e.g. "how long do I have to return an item?") and confirm the agent gets it right using retrieved context. Then comment out the retrieval call and ask again — it should get noticeably worse.

# Week 4 Answer Key — Vector Databases & RAG Pipelines

Guide: [`../../week-04-vector-db-rag.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/week-04-vector-db-rag.html)
Starter: [`../../starter-code/week-04-vector-db-rag`](../../starter-code/week-04-vector-db-rag)

This is the **fully solved** version of Week 4 — everything a student
would build is already filled in, for SME/instructor verification
against a working reference. It's not what students should be given.

## What's solved here (vs. the starter)

1. `backend/retrieval.py` — `retrieve()` embeds the query (`task_type="RETRIEVAL_QUERY"`), queries the local Chroma `documents` collection for the k nearest chunks, and returns their text. The starter leaves this raising `NotImplementedError`.
2. `backend/agent.py` — `run_agent()` calls `retrieve()`, joins the chunks into a `system_instruction`, and passes it on both `interactions.create()` calls (the initial one and the one inside the tool-call loop). The starter leaves this commented out.

Weeks 1-2's agent loop and tool calling, and `backend/ingest.py`, were already complete in the starter and are unchanged here.

## Run it

No database setup step this week — Chroma creates `backend/chroma_db/` the first time `ingest.py` runs, right there on disk.

**Recommended: GitHub Codespaces.** Push this folder to its own repo, then **Code → Codespaces → Create codespace on main** — `.devcontainer/devcontainer.json` installs `backend/requirements.txt` and creates `backend/.env` automatically. Add your `GEMINI_API_KEY` to `backend/.env`, then:
```bash
cd backend
python ingest.py          # load the sample docs into your local vector DB
uvicorn main:app --reload
```

**Running locally instead?**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/python ingest.py
./venv/bin/uvicorn main:app --reload
```

Either way, ask something that can only be answered from the sample docs (e.g. "how long do I have to return an item?") and confirm the agent gets it right using retrieved context. Then comment out the retrieval call and ask again — it should get noticeably worse.

# HOS 2 Answer Key — Embeddings, Chunking & RAG Pipelines

Guide: [`../../hos-02-embeddings-rag.html`](../../hos-02-embeddings-rag.html)
Starter: [`../../starter-code/hos-02-embeddings-rag`](../../starter-code/hos-02-embeddings-rag)

Reference solution for SME/instructor verification, built on top of HOS 1's tool-calling app (`tools.py`, `main.py`, `frontend/` carried forward unchanged).

## What's here

- `backend/ingest.py` — chunks every `.md` file in `sample_docs/` and inserts embeddings into a local Chroma `documents` collection.
- `backend/retrieval.py` — embeds the query and returns the k nearest chunks.
- `backend/agent.py` — HOS 1's tool-calling loop, extended to inject retrieved chunks as a system instruction on every call.
- `backend/sample_docs/exchanges.md` — a sixth sample doc, added **by hand** (no AI assistance) as this HOS's Understand & Refine — it wasn't part of the original five and proves the full ingest → embed → retrieve path was actually understood, not just copy-pasted.
- `EVALUATE.md` / `ANALYZE.md` — reference notes for the two written stages.

## Run it

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/python ingest.py          # embeds all 6 sample docs, including exchanges.md
./venv/bin/uvicorn main:app --reload
```

In a second terminal:

```bash
cd frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

Ask "how long do I have to exchange an item?" — a correct answer citing **45 days** confirms `exchanges.md` was actually picked up by retrieval, not just sitting unused in the folder.

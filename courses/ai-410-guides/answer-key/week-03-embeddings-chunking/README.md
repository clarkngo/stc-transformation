# Week 3 Answer Key — Embeddings & Chunking

Guide: [`../../week-03-embeddings-chunking.html`](../../week-03-embeddings-chunking.html)
Starter: [`../../starter-code/week-03-embeddings-chunking`](../../starter-code/week-03-embeddings-chunking)

Reference solution for SME/instructor verification — not what students should be given. Not part of the main chat app — a standalone sandbox script.

## What's solved here (vs. the starter)

`sandbox/ingest_sandbox.py`'s `chunk_text()` — splits on paragraphs first (natural semantic boundaries) rather than a flat word-count split, only breaking a paragraph further if it exceeds `chunk_size` words, with `overlap` words carried into the next piece. The starter ships only the flat word-count fallback described in the TODO as a placeholder to replace. `embed_chunks()` was already solved in the starter and is unchanged.

## Run it

```bash
cd sandbox
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/python ingest_sandbox.py
```

Check the printed top-3 chunks for the sample query — the top result should genuinely be the most relevant chunk (the return-policy document), not just the first one alphabetically.

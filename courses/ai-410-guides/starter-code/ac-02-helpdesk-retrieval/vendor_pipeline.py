"""
The help-desk team's retrieval pipeline — someone else's code.
It runs without errors. Don't edit it: your job is to measure it.

    chunks = vendor_chunks()                       # how they split the articles
    index  = build_index(chunks, LocalEmbedder())  # embed every chunk
    search(index, "how do I reset my password?", k=1)
      -> [{"source": "accounts-passwords", "text": "...", "score": 0.41}]
"""

import pathlib

from embedders import similarity

KB_DIR = pathlib.Path(__file__).parent / "kb"
CHUNK_CHARS = 500  # split every 500 characters


def load_articles() -> dict:
    """article_id -> full markdown text, for every file in kb/."""
    return {p.stem: p.read_text() for p in sorted(KB_DIR.glob("*.md"))}


def vendor_chunks() -> list[dict]:
    """The team's chunking: cut each article every CHUNK_CHARS characters."""
    chunks = []
    for article_id, text in load_articles().items():
        for i in range(0, len(text), CHUNK_CHARS):
            chunks.append({"source": article_id, "text": text[i:i + CHUNK_CHARS]})
    return chunks


def build_index(chunks: list[dict], embedder) -> list[dict]:
    """Embed every chunk once. Returns the chunks with a "vector" added."""
    return [{**c, "vector": embedder.embed_document(c["text"])} for c in chunks]


def search(index: list[dict], question: str, embedder, k: int = 1) -> list[dict]:
    """The k most similar chunks to the question, best first."""
    q = embedder.embed_query(question)
    scored = [
        {"source": c["source"], "text": c["text"], "score": round(similarity(q, c["vector"]), 4)}
        for c in index
    ]
    return sorted(scored, key=lambda c: c["score"], reverse=True)[:k]

"""
Week 3 sandbox: chunk the sample docs, embed them, and sanity-check
similarity scores. This script is NOT wired into the chat app — it's
a place to build intuition before Week 4 connects things for real.

Run: python ingest_sandbox.py
"""

import glob
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from the environment
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024  # gemini-embedding-001 defaults to 3072; we ask for a smaller size
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into ~chunk_size-word pieces with a little overlap so a
    chunk boundary doesn't cut a sentence's meaning in half.

    TODO(week3): implement this. A reasonable approach:
      1. Split on paragraphs first (they're natural semantic boundaries).
      2. If a paragraph is longer than chunk_size words, split it further
         by word count, carrying `overlap` words from the end of one
         chunk into the start of the next.

    Starter approach (word-count only, ignores paragraphs) — replace
    this with something better:
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap  # step back by `overlap` words
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Call the Gemini embeddings endpoint on a list of chunks and
    return one vector per chunk, in the same order.

    TODO(week3): implement this using the genai client, e.g.:

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=chunks,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM,
            ),
        )
        return [e.values for e in result.embeddings]
    """
    raise NotImplementedError("implement embed_chunks() — see the TODO above")


def embed_query(query: str) -> list[float]:
    """Same embedding call, but for a single query string at search time."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return result.embeddings[0].values


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_sample_docs() -> list[str]:
    texts = []
    for path in sorted(glob.glob(os.path.join(SAMPLE_DOCS_DIR, "*.md"))):
        with open(path) as f:
            texts.append(f.read())
    return texts


def main():
    docs = load_sample_docs()
    print(f"Loaded {len(docs)} sample documents.")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    print(f"Produced {len(all_chunks)} chunks.")

    vectors = embed_chunks(all_chunks)
    print(f"Embedded {len(vectors)} chunks.")

    # A quick retrieval test — try a few different queries and see if
    # the top result is genuinely the most relevant chunk, by eye.
    query = "How long do I have to return an item?"
    q_vec = embed_query(query)

    scored = sorted(
        zip(all_chunks, vectors),
        key=lambda pair: cosine_sim(q_vec, pair[1]),
        reverse=True,
    )

    print(f"\nTop 3 chunks for query: {query!r}\n")
    for chunk, vector in scored[:3]:
        score = cosine_sim(q_vec, vector)
        print(f"[{score:.3f}] {chunk[:120]}...\n")


if __name__ == "__main__":
    main()

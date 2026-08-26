"""
Loads the sample docs, chunks + embeds them, and inserts them into the
documents table in Supabase — solved in Week 4, unchanged here.
"""

import glob
import os

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [e.values for e in result.embeddings]


def main():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    total = 0
    for path in sorted(glob.glob(os.path.join(SAMPLE_DOCS_DIR, "*.md"))):
        with open(path) as f:
            text = f.read()

        chunks = chunk_text(text)
        vectors = embed_chunks(chunks)

        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                "insert into documents (content, embedding, source) values (%s, %s, %s)",
                (chunk, vector, os.path.basename(path)),
            )
        total += len(chunks)
        print(f"Inserted {len(chunks)} chunks from {os.path.basename(path)}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone — {total} chunks inserted.")


if __name__ == "__main__":
    main()

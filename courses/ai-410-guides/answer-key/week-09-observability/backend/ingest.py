"""
Loads the sample docs, chunks + embeds them, and inserts them into the
documents collection in a local Chroma database — solved in Week 4, unchanged here.

Run this once before starting the server:  python ingest.py
"""

import glob
import os

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


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
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(name="documents")

    total = 0
    for path in sorted(glob.glob(os.path.join(SAMPLE_DOCS_DIR, "*.md"))):
        with open(path) as f:
            text = f.read()

        chunks = chunk_text(text)
        vectors = embed_chunks(chunks)
        source = os.path.basename(path)

        collection.add(
            ids=[f"{source}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=chunks,
            metadatas=[{"source": source} for _ in chunks],
        )
        total += len(chunks)
        print(f"Inserted {len(chunks)} chunks from {source}")

    print(f"\nDone — {total} chunks inserted.")


if __name__ == "__main__":
    main()

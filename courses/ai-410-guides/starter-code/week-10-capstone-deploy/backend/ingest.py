"""
Loads the sample docs, chunks + embeds them, and inserts them into the
documents collection in Chroma — solved in Week 4, unchanged here except
that Chroma now runs as its own service (see CHROMA_HOST/CHROMA_PORT),
since the API and worker are separate deployed services this week.
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
    chroma = chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_PORT", 8001)),
    )
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

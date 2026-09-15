"""
Job functions the worker runs. Self-contained (not importing from
ingest.py) since this module runs in a separate worker process.
"""

import os

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # main.py imports this module before calling load_dotenv()
                # itself, so this module needs its own env vars loaded
                # before constructing the client below.
client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024
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


def ingest_document_job(doc_path: str) -> dict:
    with open(doc_path) as f:
        text = f.read()

    chunks = chunk_text(text)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )
    vectors = [e.values for e in result.embeddings]

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(name="documents")
    source = os.path.basename(doc_path)
    collection.add(
        ids=[f"{source}-{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": source} for _ in chunks],
    )

    return {"chunks_inserted": len(chunks)}

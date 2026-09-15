"""
Chunk and embed the sample docs into a local Chroma collection.
Run once (or whenever sample_docs/ changes): python ingest.py
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
EMBED_DIM = 1024  # must match retrieval.py's output_dimensionality
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


def main():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(name="documents")

    for doc_path in glob.glob(os.path.join(os.path.dirname(__file__), "sample_docs", "*.md")):
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

        source = os.path.basename(doc_path)
        collection.add(
            ids=[f"{source}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=chunks,
            metadatas=[{"source": source} for _ in chunks],
        )
        print(f"Inserted {len(chunks)} chunks from {source}")


if __name__ == "__main__":
    main()

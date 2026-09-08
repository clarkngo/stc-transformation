"""
Retrieval for the RAG pipeline: embed the incoming query, then find
the closest chunks in the documents collection by vector distance.
"""

import os

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024  # must match output_dimensionality in ingest.py
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Return the top-k document chunks most relevant to `query`.

    TODO(week4): implement this.
      1. Embed the query with the SAME model and dimensionality used in
         ingest.py — but task_type="RETRIEVAL_QUERY" this time, not
         "RETRIEVAL_DOCUMENT" (Gemini optimizes the vector differently
         depending on which side of the search it's used for).
      2. Open the same local Chroma collection ingest.py wrote to, and
         query it for the k nearest chunks to that vector.
      3. Return just the chunk text for each result.

    Starter shape:

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBED_DIM,
            ),
        )
        q_vec = result.embeddings[0].values

        chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma.get_or_create_collection(name="documents")
        results = collection.query(query_embeddings=[q_vec], n_results=k)
        return results["documents"][0]
    """
    raise NotImplementedError("implement retrieve() — see the TODO above")

"""
Retrieval for the RAG pipeline: embed the incoming query, then find
the closest chunks in the documents table by vector distance.
"""

import os

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024  # must match the vector(1024) column in schema.sql and ingest.py


def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Return the top-k document chunks most relevant to `query`.

    TODO(week4): implement this.
      1. Embed the query with the SAME model and dimensionality used in
         ingest.py — but task_type="RETRIEVAL_QUERY" this time, not
         "RETRIEVAL_DOCUMENT" (Gemini optimizes the vector differently
         depending on which side of the search it's used for).
      2. Query the documents table using pgvector's `<->` distance
         operator, ordered ascending (closer = smaller distance),
         limited to k rows.
      3. Return just the chunk text for each row.

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

        conn = psycopg.connect(os.environ["DATABASE_URL"])
        rows = conn.execute(
            "select content from documents order by embedding <-> %s limit %s",
            (q_vec, k),
        ).fetchall()
        return [row[0] for row in rows]
    """
    raise NotImplementedError("implement retrieve() — see the TODO above")

"""
Retrieval for the RAG pipeline: embed the incoming query, then find
the closest chunks in the documents table by vector distance.
"""

import os

import psycopg
import voyageai
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()


def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Return the top-k document chunks most relevant to `query`.

    TODO(week4): implement this.
      1. Embed the query with the SAME model used in ingest.py
         (voyage-3, input_type="query" this time, not "document").
      2. Query the documents table using pgvector's `<->` distance
         operator, ordered ascending (closer = smaller distance),
         limited to k rows.
      3. Return just the chunk text for each row.

    Starter shape:

        q_vec = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
        conn = psycopg.connect(os.environ["DATABASE_URL"])
        rows = conn.execute(
            "select content from documents order by embedding <-> %s limit %s",
            (q_vec, k),
        ).fetchall()
        return [row[0] for row in rows]
    """
    raise NotImplementedError("implement retrieve() — see the TODO above")

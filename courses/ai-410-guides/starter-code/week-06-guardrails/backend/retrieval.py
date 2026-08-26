"""
Retrieval for the RAG pipeline — solved in Week 4, unchanged here.
"""

import os

import psycopg
import voyageai
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()


def retrieve(query: str, k: int = 5) -> list[str]:
    q_vec = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    rows = conn.execute(
        "select content from documents order by embedding <-> %s limit %s",
        (q_vec, k),
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]

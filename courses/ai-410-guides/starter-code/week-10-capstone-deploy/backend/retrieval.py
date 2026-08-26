"""
Retrieval for the RAG pipeline. Instrumented (Week 9, solved).
"""

import os

import psycopg
import voyageai
from dotenv import load_dotenv
from langfuse.decorators import observe

load_dotenv()

vo = voyageai.Client()


@observe()
def retrieve(query: str, k: int = 5) -> list[str]:
    q_vec = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    rows = conn.execute(
        "select content from documents order by embedding <-> %s limit %s",
        (q_vec, k),
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]

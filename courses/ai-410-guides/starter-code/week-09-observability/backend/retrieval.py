"""
Retrieval for the RAG pipeline — solved in Week 4. Your Week 9 job:
trace this as its own step, separate from the main agent call, so it
shows up as a distinct span in the dashboard.
"""

import os

import psycopg
import voyageai
from dotenv import load_dotenv

# TODO(week9): uncomment once you've set your LANGFUSE_* env vars
# from langfuse.decorators import observe

load_dotenv()

vo = voyageai.Client()


# TODO(week9): add @observe() above this function.
def retrieve(query: str, k: int = 5) -> list[str]:
    q_vec = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    rows = conn.execute(
        "select content from documents order by embedding <-> %s limit %s",
        (q_vec, k),
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]

"""
Retrieval for the RAG pipeline — solved in Week 4. Your Week 9 job:
trace this as its own step, separate from the main agent call, so it
shows up as a distinct span in the dashboard.
"""

import os

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types

# TODO(week9): uncomment once you've set your LANGFUSE_* env vars
# from langfuse.decorators import observe

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024


# TODO(week9): add @observe() above this function.
def retrieve(query: str, k: int = 5) -> list[str]:
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
    conn.close()
    return [row[0] for row in rows]

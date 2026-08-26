"""
Job functions the worker runs — ingestion (Week 7) is solved below.
Your Week 8 job: send a push notification when ingestion finishes.
See the TODO near the bottom of ingest_document_job().
"""

import os

import psycopg
from google import genai
from google.genai import types

from devices import get_push_token
from push import send_push_notification

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1024


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def ingest_document_job(doc_path: str, device_id: str | None = None) -> dict:
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

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    for chunk, vector in zip(chunks, vectors):
        cur.execute(
            "insert into documents (content, embedding, source) values (%s, %s, %s)",
            (chunk, vector, os.path.basename(doc_path)),
        )
    conn.commit()
    conn.close()

    # TODO(week8): send a push notification here so the mobile app
    # knows ingestion finished, even if it's backgrounded. You'll need
    # the device's push token (already looked up above via device_id).
    #
    #   if device_id:
    #       token = get_push_token(device_id)
    #       if token:
    #           send_push_notification(
    #               token,
    #               title="Document ready",
    #               body=f"Finished processing {os.path.basename(doc_path)}.",
    #           )

    return {"chunks_inserted": len(chunks)}

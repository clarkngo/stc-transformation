"""
Job functions the worker runs. These are plain Python functions —
they don't return an HTTP response, they just do the work.
"""

import os

import psycopg
import voyageai

vo = voyageai.Client()


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
    """
    TODO(week7): this is the function the queue will run in the
    background. Move your ingestion logic here (chunk -> embed ->
    insert), reading the file at `doc_path`:

        with open(doc_path) as f:
            text = f.read()

        chunks = chunk_text(text)
        vectors = vo.embed(chunks, model="voyage-3", input_type="document").embeddings

        conn = psycopg.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                "insert into documents (content, embedding, source) values (%s, %s, %s)",
                (chunk, vector, os.path.basename(doc_path)),
            )
        conn.commit()
        conn.close()

        return {"chunks_inserted": len(chunks)}

    Return a plain dict (not an object) — rq stores the return value
    as the job's result, which the /jobs/{id} endpoint reads.
    """
    raise NotImplementedError("implement ingest_document_job() — see the TODO above")

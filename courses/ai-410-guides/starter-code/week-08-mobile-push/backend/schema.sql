-- Run this in the Supabase SQL editor before running ingest.py.
-- The vector size (1024) must match the output_dimensionality you
-- request from gemini-embedding-001 in ingest.py/retrieval.py/jobs.py
-- — if you change one, change them all.

create extension if not exists vector;

create table if not exists documents (
    id bigserial primary key,
    content text not null,
    embedding vector(1024) not null,
    source text
);

-- Speeds up similarity search once you have more than a few hundred rows.
-- Safe to run even with only a handful of rows in it for this course.
create index if not exists documents_embedding_idx
    on documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

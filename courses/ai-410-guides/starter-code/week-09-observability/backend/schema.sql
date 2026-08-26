-- Run this in the Supabase SQL editor before running ingest.py.
-- The vector size (1024) must match voyage-3's output dimensions —
-- if you switch embedding models, update this to match.

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

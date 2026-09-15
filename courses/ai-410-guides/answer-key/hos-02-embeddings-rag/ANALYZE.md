# Reference: Analyze stage

The most natural target in this HOS is why `retrieval.py` and `ingest.py` embed with *different* `task_type` values despite calling the same `embed_content` API:

```python
# ingest.py
config=types.EmbedContentConfig(
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=EMBED_DIM,
)

# retrieval.py
config=types.EmbedContentConfig(
    task_type="RETRIEVAL_QUERY",
    output_dimensionality=EMBED_DIM,
)
```

A submission that's actually understood this should be able to explain:

1. **Why the model needs to know which side of the search a piece of text is on.** A document chunk and a user's question are different *kinds* of text even when they're about the same topic — a chunk is dense, declarative content; a query is often short and phrased as a question. Gemini's embedding model optimizes the vector differently depending on which role you tell it the text is playing, so the two ends of the same search aren't treated identically.

2. **Why `output_dimensionality` has to match exactly (1024 in both files) even though `task_type` differs.** Chroma compares vectors by distance — that only makes sense if the query vector and the stored document vectors live in the same-dimensional space. If `EMBED_DIM` drifted between the two files, `collection.query()` would either error or silently return meaningless results.

3. **What would break if `ingest.py` also used `RETRIEVAL_QUERY`.** Not a crash — both are valid, same-dimension vectors. The chunks would just be embedded slightly worse for their actual role (being searched *against*), typically showing up as retrieval that "sort of" works but pulls back less relevant chunks than it should, without ever producing an obvious error to point to.

A write-up that says "they're embedded with different settings" hasn't analyzed this — the bar is explaining why the search would still *look* like it works with the wrong setting, just worse, which is exactly the kind of bug that's easy to miss without understanding the mechanism.

# Reference: Analyze stage

Two natural targets in this HOS — a strong submission goes deep on one, not both shallowly.

## Option A: why `call_with_guardrail` takes a `correction_hint`, not just a retry

```python
for attempt in range(max_retries + 1):
    try:
        raw = fn(*args, **kwargs)
        return schema.model_validate(raw)
    except ValidationError as e:
        last_error = e
        if correction_hint and attempt < max_retries:
            kwargs = correction_hint(str(e), *args, **kwargs) or kwargs
```

A plain retry re-runs the exact same call and, against a nondeterministic model, has a real chance of getting the same malformed result again — the model doesn't know it was wrong. `correction_hint` exists so the *next* attempt can be told what specifically failed (via the `ValidationError`'s message), giving the retry an actual chance of succeeding instead of just rolling the dice again. A submission that's understood this should be able to explain why "just try again" and "try again with the error fed back in" are meaningfully different strategies.

## Option B: why the worker has to be a separate process from the API

`main.py`'s `/documents` endpoint calls `queue.enqueue(ingest_document_job, ...)` and returns immediately — it never calls `ingest_document_job` itself. The actual chunking/embedding work only happens when `worker.py`, running as its own process, pulls that job off the Redis queue and executes it. A submission that's understood this should be able to explain: why `uvicorn main:app` alone is not enough (the API process only ever *enqueues*, it never *runs* jobs); what `redis_conn` is actually doing (it's the shared communication channel — the API writes a job description to it, the worker reads jobs from it, and they don't need to know about each other directly); and what "queued forever" actually means diagnostically (the job description is sitting in Redis, correctly, with nothing consuming it).

A write-up that says "the worker runs the jobs" for either option hasn't analyzed anything — the bar is explaining the failure mode each design choice prevents.

# Reference: Evaluate stage

## Things a typical AI-scaffolded first pass gets right
- `call_with_guardrail()` retries with the validation error visible in its log line, rather than failing silently.
- `/documents` returns a `job_id` immediately instead of blocking until ingestion finishes.

## Things worth specifically testing
- **A malformed calculator call** — a message like "calculate the square root of a banana" tends to produce arguments the model itself isn't confident in. Does the guardrail catch a bad `expression` and retry, or does the whole request 500?
- **Chat responsiveness during ingestion** — start a document upload, then immediately send a normal chat message. Does the chat reply promptly (proving the job is genuinely running in the background), or does it hang until ingestion finishes (a sign ingestion is still happening inline, or the worker isn't running)?
- **The worker not running** — stop `worker.py` and upload a document. `/jobs/{id}` should report `"queued"` forever, never `"finished"`. This is the single most common "it doesn't work" report for this HOS, and it's not a bug — the fix is starting `worker.py` in its own terminal.
- **The by-hand `WordCountArgs` guardrail** — send `word_count` an empty or whitespace-only string. Does it get caught and retried, or does the tool just report "0 words" as if that were a normal answer?

A submission that only shows a successful upload → finished job hasn't evaluated the async piece — the responsiveness-during-ingestion test is what actually proves it's asynchronous, not just successful.

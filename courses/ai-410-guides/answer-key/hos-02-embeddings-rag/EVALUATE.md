# Reference: Evaluate stage

## Things a typical AI-scaffolded first pass gets right
- `ingest.py` chunks and embeds `sample_docs/` into a local Chroma collection without needing an account or connection string.
- `retrieval.py` embeds the query with `task_type="RETRIEVAL_QUERY"` (not `RETRIEVAL_DOCUMENT"` — a detail an AI assistant sometimes gets wrong by using the same task type on both sides).

## A real bug worth watching for
A naive merge of "use this context, say you don't know if it's not there" with the HOS 1 tool-calling loop **breaks tool calling** — the model reads the instruction as applying to every question, including ones that need `calculate` or `word_count`, and refuses instead of calling the tool. This answer key's `agent.py` fixes it by explicitly scoping the "say you don't know" constraint to policy/product questions and telling the model the restriction doesn't apply to its tools. If a student's submission has the original broad wording, ask them to test a question that needs a tool ("what's 25 times 4?") — it's a near-certain way to surface this.

## Things worth specifically testing
- **A question only answerable from the sample docs** (e.g. "how long do I have to return an item?") — does the answer cite the actual 30-day figure from `return-policy.md`, or a plausible-sounding guess?
- **The same question with retrieval effectively disabled** (temporarily set `k=0` or comment out the retrieval call) — does the answer get noticeably worse or vaguer? If not, retrieval isn't actually contributing anything, which is worth flagging as a finding in its own right, not a failure to hide.
- **A question that needs a tool** ("what's 25 times 4?") — does the agent still call `calculate`, or does the RAG system instruction silently suppress tool use (see the bug above)?
- **A policy-shaped question the docs don't cover** ("do you offer gift wrapping?") — does the agent say it doesn't know, or hallucinate a plausible-sounding policy? (A general-knowledge question like "what's the capital of Japan?" is a *weaker* test here — a correctly-scoped system instruction should answer that directly, not decline it, since the "say you don't know" restriction should only cover policy/product questions.)
- **A question that combines both** ("what's 15% of the expedited shipping fee?") — does the agent correctly combine retrieved context (the $15 fee) with a tool call (the percentage math)?

A submission that only confirms "RAG works" on the easy case hasn't evaluated anything — the tool-calling and uncovered-policy tests are what actually prove the pipeline is doing something, not just adding a plausible-looking system prompt.

# AI 410 — Answer Key

**Instructor/SME use only — not for students.** One folder per week, each a
**complete, fully-solved snapshot** of that week's [starter code](../starter-code/),
with every `# TODO(weekN)` filled in. Use these to verify a week's components
actually work before handing the starter code to students, or to check a
student's approach against a known-working reference.

Each week's `README.md` lists exactly what was solved (vs. the starter) and how
to run it — same layout and run instructions as the matching starter-code
folder, so anything in the [top-level starter-code README](../starter-code/README.md)
about Codespaces, local venvs, or the Week 8 mobile app applies here too.

## Production schedule (8 individually-graded weeks)

Complete — every week from the reviewed 10-week schedule that has a build task.

| Folder | Solves |
|---|---|
| `week-01-foundations/` | `main.py`'s `/chat` endpoint calling Gemini for real |
| `week-02-agentic-loop/` | `tools.py`'s `calculate` tool, registered alongside the dummy `ping` tool |
| `week-03-embeddings-chunking/` | `ingest_sandbox.py`'s paragraph-aware `chunk_text()` |
| `week-04-vector-db-rag/` | `retrieval.py`'s `retrieve()`, and wiring it into `agent.py` as a system instruction |
| `week-06-guardrails/` | `guardrails.py`'s `ToolArgs` schema, guarding the `calculate` tool's arguments |
| `week-07-async-processing/` | `jobs.py`'s `ingest_document_job()`, and `main.py`'s async `/documents` + `/jobs/{id}` endpoints |
| `week-08-mobile-push/` | `jobs.py`'s push-notification send, and `mobile/App.js` registering the device's push token |
| `week-09-observability/` | `@observe()` tracing on `agent.py`/`retrieval.py`, and `score_faithfulness()` |

**Two real bugs found and fixed while verifying these live** (present in the original `week-04`/`week-08` answer keys, and — since the guide text week-04 teaches matches it verbatim — likely in the production guide's own suggested solution too):

1. The RAG system instruction ("if the answer isn't in the context, say you don't know") was unscoped, which silently suppressed tool-calling once RAG and tools coexist (Week 4 onward). Fixed by scoping the instruction to policy/product questions and explicitly exempting tools.
2. The Week 6 guardrail was applied unconditionally to every tool call, including `ping` (which takes no arguments) — every `ping` call failed validation. It only *looked* fine because the model papered over the error and guessed "pong" from the tool's own description. Fixed by only guarding `calculate`.

Both fixes are now in every week from 04/06 onward.

## Draft HOS redesign (4 merged units)

| Folder | Solves |
|---|---|
| `hos-01-foundations-agentic-loop/` | Same as Weeks 1-2, plus a hand-written second tool for Understand & Refine |
| `hos-02-embeddings-rag/` | Same as Weeks 3-4, plus a hand-written sixth sample doc for Understand & Refine |
| `hos-03-guardrails-async/` | Same as Weeks 6-7, plus a hand-written second guardrail schema for Understand & Refine |
| `hos-04-mobile-observability/` | Same as Weeks 8-9, plus hand-written `score_faithfulness()` for Understand & Refine |

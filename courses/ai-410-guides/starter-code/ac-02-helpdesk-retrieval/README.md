# Application Challenge 2 — Riverbend College help-desk retrieval

Guide: [`../../ac-02-helpdesk-retrieval.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/ac-02-helpdesk-retrieval.html)

In HOS 2 you built retrieval. Here you **measure** someone else's, improve it, and decide when their bot should stop answering.

## What's here

| File | What it is | You edit it? |
|---|---|---|
| `evaluate.py` | `hit_rate`, your `chunk_article`, `answer_or_abstain`, and your `MIN_SCORE` | **Yes** |
| `tests/test_my_rules.py` | Your tests for your threshold | **Yes** |
| `DECISIONS.md` | Your one-page decisions memo | **Yes** |
| `REPORT.md` | The output of `python run_eval.py`, pasted | **Yes** |
| `vendor_pipeline.py` | The help desk's pipeline: chunk, embed, search | No (read it) |
| `kb/` | The six help-desk articles | No |
| `eval_set.json` | 16 answerable questions with known answers, 4 the articles can't answer | No |
| `embedders.py` | An offline word-count embedder (used by the tests) and Gemini's real one (optional) | No |
| `run_eval.py`, `tests/test_evaluate.py` | The report script and the tests your code must pass | No |

## Run it

```bash
pytest
python run_eval.py
```

No API key needed. Optional: put your Gemini key in `.env` and run `python run_eval.py --gemini` to compare with real embeddings.

See the guide for the full spec and what to submit.

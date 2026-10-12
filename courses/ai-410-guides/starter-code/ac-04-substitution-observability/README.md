# Application Challenge 4 — Puget Pantry substitution observability

Guide: [`../../ac-04-substitution-observability.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/ac-04-substitution-observability.html)

In HOS 4 you traced and scored your own chat app. Here you instrument someone else's AI feature, a grocery service that swaps out-of-stock items and pushes the change to the customer's phone, and decide what's worth paging someone for.

## What's here

| File | What it is | You edit it? |
|---|---|---|
| `instrumented.py` | `score_substitution`, `run_order` (traced), and your `should_page` | **Yes** |
| `tests/test_my_rules.py` | Your tests for your paging rule | **Yes** |
| `DECISIONS.md` | Your one-page decisions memo | **Yes** |
| `REPORT.md` | The output of `python run_day.py`, pasted | **Yes** |
| `feature.py` | The substitution feature: four steps, no tracing | No (read it) |
| `pantry_data.py` | Catalog, customers and their allergies, one day of orders | No |
| `obs.py` | A tiny in-memory tracer shaped like Langfuse: traces, spans, scores | No |
| `fake_model.py`, `push.py` | Recorded model choices and recorded push notifications | No |
| `run_day.py`, `tests/test_instrumented.py` | The day runner and the tests your code must pass | No |

## Run it

```bash
pytest
python run_day.py
```

No API key, Langfuse project, or phone needed. See the guide for the full spec and what to submit.

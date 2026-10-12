# Application Challenge 3 — Pacific Bean Coffee refund guardrail

Guide: [`../../ac-03-refund-guardrail.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/ac-03-refund-guardrail.html)

In HOS 3 you guarded a calculator. Here the tool moves money: build the guardrail for a coffee roaster's refund tool, and decide what the bot may refund without a person.

## What's here

| File | What it is | You edit it? |
|---|---|---|
| `refunds.py` | `RefundArgs`, `issue_refund`, and your `AUTO_REFUND_LIMIT` | **Yes** |
| `tests/test_my_rules.py` | Your tests for your routing rules | **Yes** |
| `DECISIONS.md` | Your one-page decisions memo | **Yes** |
| `REPORT.md` | The output of `python run_cases.py`, pasted | **Yes** |
| `store_data.py` | Orders, customers, the refund window, the $500 hard limit, a fixed clock | No (read it) |
| `review_queue.py` | The human review queue (a stand-in for HOS 3's Redis queue) | No |
| `recorded_calls.json` | 12 tool calls the model really made in testing, good and bad | No |
| `run_cases.py`, `tests/test_refunds.py` | The replay script and the tests your code must pass | No |

## Run it

```bash
pytest
python run_cases.py
```

No API key needed. See the guide for the full spec and what to submit.

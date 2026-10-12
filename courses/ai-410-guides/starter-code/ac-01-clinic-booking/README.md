# Application Challenge 1 — Cascade Family Clinic booking agent

Guide: [`../../ac-01-clinic-booking.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/ac-01-clinic-booking.html)

You built an agent with a calculator in HOS 1. Here you build the tools for a different business, from a spec, on your own.

## What's here

| File | What it is | You edit it? |
|---|---|---|
| `tools.py` | Three tool functions to write, plus their schemas | **Yes** |
| `tests/test_my_rules.py` | Your tests for your late-cancellation rule | **Yes** |
| `DECISIONS.md` | Your one-page decisions memo | **Yes** |
| `TRANSCRIPT.md` | Your two chat sessions, pasted | **Yes** |
| `clinic_data.py` | The clinic's slots, patients, and fixed clock | No (read it) |
| `agent.py` | The chat app — HOS 1's loop, already written | No |
| `tests/test_tools.py` | The tests your tools must pass | No |

## Run it

```bash
pytest
```

The tests need no API key. Once they pass, put your free Gemini key in `.env` and try the agent:

```bash
python agent.py --patient P1001
```

See the guide for the full spec, the three messages to put in your transcript, and what to submit.

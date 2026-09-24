# AI 420 — HOS 3: Reasoning Traces, Failure Diagnosis & Self-Correction

Guide: https://clarkngo.github.io/stc-transformation/courses/ai-420-guides/hos-03-traces-self-correction.html

**Don't read `support_agent/` yet.** This agent fails in several ways, and the point is to find each one from the traces first.

**Already working**
- `run_scenarios.py`: runs the six scenarios in `scenarios.json`, prints a trace of each, and saves it to `traces/`.
- `support_agent/`: Harbor Supply Co.'s customer-support agent (plus `adk web` if you want to poke at it).

**You build**
- Stage 1: Langfuse tracing (a `tracing.py` module that `run_scenarios.py` picks up). You write the prompt.
- Stage 4: the fixes, plus `support_agent/guards.py`, by hand.

```bash
cp .env.example .env        # paste your GOOGLE_API_KEY (and Langfuse keys for Stage 1)
python data/seed_db.py      # Codespaces does this for you
python run_scenarios.py
```

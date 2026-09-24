# AI 420 — HOS 4: Multi-Agent Orchestration & Agent-on-Agent Evaluation

Guide: https://clarkngo.github.io/stc-transformation/courses/ai-420-guides/hos-04-multi-agent-evaluation.html

**Already working**
- `single_agent/`: the repaired support agent from HOS 3, with its guards. This is your baseline.
- `evals/scenarios.json`: 12 test scenarios, each a persona, a goal, and a success rule.
- `evals/pricing.json`: example token prices. Update them to your model's current list price before reporting cost.

**You build** (goal only; you design it)
- Stage 1: `team_agent/` (a coordinator + specialists) and `evals/run_eval.py` (a simulated customer, a judge, and metrics: success rate, cost per task, p50/p95 latency), plus a side-by-side report.
- Stage 4: `evals/agreement.py`, by hand: how often does the judge agree with you?

```bash
cp .env.example .env        # paste your GOOGLE_API_KEY
python data/seed_db.py      # Codespaces does this for you
adk web                     # try single_agent by hand first
```

# AI 420 — Faculty smoke tests (instructor/SME only)

Checks every HOS **without a Gemini key**, so you can confirm an environment works before spending free-tier quota.

```bash
pip install -r ../starter-code/portfolio-template/requirements.txt   # once; Codespaces built from the portfolio template already has this
bash run_all.sh            # answer keys
bash run_all.sh starter    # starter code
```

What it checks, per HOS:

| HOS | Real services | Agent logic (scripted stand-in model) |
|---|---|---|
| 1 | Wikipedia search + intros, calculator | Hand-built ReAct loop, unknown-tool errors, step budget, plan-and-execute, baseline, ADK agent loads, missing-key message |
| 2 | Open-Meteo, a live web page (trafilatura), SQLite (reads, refused writes), Chroma memory | MCP server over stdio, ADK `McpToolset` discovery, agent → MCP tool round trip, session-state memory |
| 3 | SQLite | `run_scenarios.py` end to end: the five planted failures reproduce in the starter and recover in the answer key; guards; traces/ output; `tracing.py` without Langfuse keys |
| 4 | SQLite | Single agent, coordinator → specialist handoff, the eval harness (simulated user → agent → judge), metric math, `compare.py`, `agreement.py` |

What it can't check: how a **real** Gemini model behaves. That's the faculty testing pass — see the
[Faculty Test Runbook](https://clarkngo.github.io/stc-transformation/courses/ai-420-guides/faculty-test-runbook.html).

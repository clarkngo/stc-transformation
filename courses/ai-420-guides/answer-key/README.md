# AI 420 — Answer Keys (instructor/SME only)

Fully solved reference for each of the 4 HOS. Use it to verify the starter code and guides
actually work, not to hand to students.

| HOS | What the answer key adds on top of the starter |
|---|---|
| `hos-01-agent-anatomy/` | `research/react_loop.py` (hand-built ReAct loop), `research/plan_execute.py` (Stage 4) |
| `hos-02-tools-mcp-memory/` | Browser + read-only database tools on the MCP server, Stage 4's `order_status` / `low_stock_report`, session-state + Chroma memory (`ops_agent/memory.py`) |
| `hos-03-traces-self-correction/` | `tracing.py` (Langfuse), the five fixes in `support_agent/tools.py`, `support_agent/guards.py` (Stage 4), `FAILURE_LOG.md` (reference Stage 2 log) |
| `hos-04-multi-agent-evaluation/` | `team_agent/` (coordinator + 3 specialists), `evals/run_eval.py`, `evals/compare.py`, `evals/agreement.py` (Stage 4) |

## Verification status — September 24, 2026

**Checked offline** (no Gemini key used):

- All four `requirements.txt` files install cleanly into fresh Python 3.12 environments, and `pip check` passes. Versions resolved: `google-adk` 2.9.2, `mcp` 2.2.0, `chromadb` 1.5.9, `opentelemetry-sdk` 1.42.1.
- **Every tool against the real free services:** Wikipedia (search + intro extracts), Open-Meteo (geocoding, forecast, marine), a live web page through trafilatura, SQLite (including rejected writes, multi-statement SQL, and bad columns), and Chroma memory (save, recall, customer scoping).
- **HOS 1 task answers** are in the Wikipedia intros the tools return (expected answers in `tasks.json` were adjusted to match those intros).
- **Agent logic against a scripted stand-in model:** the hand-built loop (tool calls, unknown tools, step budget), ADK agents loading, MCP tools discovered and called over stdio through `McpToolset`, HOS 3's five planted failures each reproducing as designed in the starter and recovering in the answer key, and HOS 4's harness (single agent and the coordinator → specialist handoff, metric math, compare and agreement reports).

## Re-check — September 30, 2026

- Fresh installs on Python 3.12, per HOS and via the portfolio template's single `requirements.txt`: same versions, `pip check` clean.
- The offline checks above, now committed as `../faculty-smoke-tests/` (`bash run_all.sh` and `bash run_all.sh starter`): all four HOS pass for both the answer keys and the starters. `adk web` starts in each HOS folder and lists the expected agents.
- **Fixed:** in a portfolio-repo Codespace (the path the guides tell students to use), nothing built `data/harbor.db` for HOS 2–4, because only the per-HOS devcontainers seed it. The MCP server (HOS 2) and the support tools (HOS 3–4) now build it the first time they're imported, in both the starters and the answer keys.

**Not yet checked:** end-to-end runs against Gemini (`gemini-3.6-flash`). That's the faculty testing pass: how real models behave on each scenario, actual time per HOS, and free-tier rate limits at class scale. Step by step: the [Faculty Test Runbook](../faculty-test-runbook.html).

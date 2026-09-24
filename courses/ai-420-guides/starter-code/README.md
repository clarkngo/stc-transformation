# AI 420 — Starter Code

One folder per HOS, matching the [4 hands-on guides](https://clarkngo.github.io/stc-transformation/courses/ai-420-guides/index.html),
plus a portfolio template. Each HOS folder is a complete, standalone snapshot: copy it anywhere and it runs on its own.

| Folder | What's already working | What you build |
|---|---|---|
| `portfolio-template/` | A repo skeleton whose root devcontainer installs everything all four HOS need | Create your portfolio repo from this in Week 1 |
| `hos-01-agent-anatomy/` | Wikipedia + calculator tools, a single-call baseline, 5 test questions, a comparison script, the same agent in ADK | A hand-built agent loop, then plan-and-execute |
| `hos-02-tools-mcp-memory/` | A store database, an MCP server with one API tool, an ADK agent connected to it | Browser and database tools, short- and long-term memory |
| `hos-03-traces-self-correction/` | A support agent that fails in several ways, 6 scenarios, a trace runner | Langfuse tracing, a failure log, fixes, self-correction guards |
| `hos-04-multi-agent-evaluation/` | The repaired single agent, 12 evaluation scenarios, a price sheet | A multi-agent team, an agent-on-agent evaluation harness, a judge-agreement check |

## Setup (every HOS)

1. Copy the HOS folder into your portfolio repo and open the repo in GitHub Codespaces.
   (Codespaces only reads a `.devcontainer/` at the repo root. The portfolio template's covers all four HOS.
   Each HOS folder also has its own, for opening it as a repo by itself.)
2. In the HOS folder: `cp .env.example .env`, then paste your free Gemini API key as `GOOGLE_API_KEY`. Never commit `.env`.
3. HOS 2–4 build their database automatically in Codespaces. Anywhere else, run `python data/seed_db.py` once.
4. Read that HOS's guide before you start. It says exactly what to build and submit.

Everything here runs on free tiers with no credit card: the Gemini API free tier, Wikipedia, Open-Meteo, local SQLite and Chroma, and Langfuse's free cloud plan.

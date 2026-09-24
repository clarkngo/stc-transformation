# My Agent Portfolio — AI 420 Agentic AI

<!-- Replace this paragraph with two sentences about you and what this portfolio shows. -->

Each folder is one working agent built during AI 420. Every one runs in GitHub Codespaces on free tiers.

| # | Agent | What it does | Skills shown |
|---|---|---|---|
| 01 | [Research agent](hos-01-agent-anatomy/) | Answers multi-step questions from Wikipedia with a hand-built agent loop, then the same agent in Google ADK | Agent loop, ReAct vs. plan-and-execute |
| 02 | [Operations agent](hos-02-tools-mcp-memory/) | Runs a marine supply store's lookups through its own MCP tool server, with long-term memory | Custom tools, MCP, memory |
| 03 | [Self-correcting support agent](hos-03-traces-self-correction/) | A broken agent, diagnosed from traces and repaired to recover on its own | Trace analysis, self-correction |
| 04 | [Support team + evaluation](hos-04-multi-agent-evaluation/) | A coordinator with specialist agents, tested by a simulated customer and an LLM judge | Multi-agent orchestration, agent evaluation |
| ★ | Capstone (Weeks 7–9) | | |

## Running any agent

1. Open this repo in a Codespace (the green **Code** button → **Codespaces**). Dependencies install automatically.
2. `cd` into an agent's folder, then `cp .env.example .env` and paste your free Gemini API key.
3. Follow that folder's `README.md`.

## Each agent's README should have

- What it does, in two sentences
- A screenshot of a trace (from `adk web` or Langfuse)
- One thing that went wrong while building it, and how you found and fixed it

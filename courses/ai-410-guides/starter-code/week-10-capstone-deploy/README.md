# Week 10 Starter — Capstone Integration, Deployment & Demo

Guide: [`../../week-10-capstone-deploy.html`](../../week-10-capstone-deploy.html)

## What's already working

Everything — Weeks 1-9 are all solved in this snapshot: chat, tool calling, RAG, guardrails, async ingestion, push notifications, and Langfuse tracing.

- `backend/Dockerfile` — packages the API (and, with a different start command, the worker)
- `backend/render.yaml` — deploys both as separate services on Render's free tier
- `backend/.dockerignore`

## What you'll build this week

There's no new feature to build — this week is integration and deployment:

1. Run a full local pass through the whole pipeline first and fix anything that regressed.
2. Set every environment variable from `.env` in your host's dashboard (never commit `.env`).
3. Deploy `backend/` using `render.yaml` (or adapt it for another host) — as **two** services, API and worker.
4. Update `mobile/src/api.js`'s `API_BASE` to your deployed URL (see the `# TODO(week10)` comment) and confirm push notifications still work against the live server.
5. Re-run your Week 5 bottleneck list against the deployed version.
6. Prepare and give a 5-minute live demo.

## Deploy

```bash
# In the Render dashboard: "New +" -> "Blueprint" -> point at this
# backend/ folder (containing render.yaml). Set the env var values
# for both services when prompted.
```

Free-tier hosts often spin down when idle — the first request after idle can be slow. Mention this in your demo rather than being caught off guard.

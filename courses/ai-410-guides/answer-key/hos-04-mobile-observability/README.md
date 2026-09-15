# HOS 4 Answer Key — Mobile Client, Push & Observability

Guide: [`../../hos-04-mobile-observability.html`](../../hos-04-mobile-observability.html)
Starter: [`../../starter-code/hos-04-mobile-observability`](../../starter-code/hos-04-mobile-observability)

Reference solution, built on top of HOS 3's guardrailed, async-ingesting RAG app.

## What's here

- `backend/devices.py`, `push.py` — device/push-token store and Expo push sending (reused as-is from the mobile/push pattern established earlier in the course).
- `backend/jobs.py` — ingestion (HOS 3) plus a push notification on completion.
- `backend/main.py` — adds `POST /register-device`.
- `backend/retrieval.py`, `agent.py` — traced with Langfuse's `@observe()`, plus `score_faithfulness()` — written **by hand** for Understand & Refine — a lexical-overlap heuristic scoring how much of an answer is actually grounded in the retrieved context, logged against every trace.
- `mobile/` — the Expo (React Native) client: chat screen, push-permission request, device registration.
- `EVALUATE.md` / `ANALYZE.md` — reference notes for the two written stages.

## Run it

Needs Redis (from HOS 3) plus a free [Langfuse](https://cloud.langfuse.com) project. The app works without Langfuse keys set — tracing just disables itself with a log line — but you won't see anything in a dashboard until they're configured.

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY, REDIS_URL, LANGFUSE_*
./venv/bin/python ingest.py
```

Then two separate terminals (in Codespaces, a new tab always starts back at the repo root, so `cd backend` again in each one):

```bash
# Terminal 1 — API
cd backend
./venv/bin/uvicorn main:app --reload
```
```bash
# Terminal 2 — worker
cd backend
./venv/bin/python worker.py
```

Frontend (terminal 3) same as prior HOS units. Mobile (terminal 4):

```bash
cd mobile
npm install
npx expo start
```

Push notifications need one extra one-time step: run `npx eas init` inside `mobile/` (free Expo account, no credit card — see the [Account & Service Setup Guide](../../account-setup-guide.html#expo)). `getExpoPushTokenAsync()` requires an EAS project ID as of SDK 49+; without it, `App.js` now logs a warning and returns instead of crashing — chat still works fine either way.

Verified this session: `/health`, RAG-grounded chat, tool calling, `/register-device`, and a full async-ingestion-with-push job — all live, with Langfuse keys unset (confirmed it disables gracefully rather than crashing). `score_faithfulness()` unit-tested against a grounded answer (0.75), a deliberately invented one (0.077), and a plain "I don't know" (1.0 — nothing to check). `mobile/App.js` was code-reviewed (not run live against a real device this session) — the missing-projectId fix is based on Expo's documented SDK 49+ requirement, not a live repro.

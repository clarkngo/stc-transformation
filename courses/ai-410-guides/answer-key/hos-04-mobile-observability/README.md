# HOS 4 Answer Key — Mobile Client, Push & Observability

Guide: [`../../hos-04-mobile-observability.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/hos-04-mobile-observability.html)
Starter: [`../../starter-code/hos-04-mobile-observability`](../../starter-code/hos-04-mobile-observability)

Reference solution, built on top of HOS 3's guardrailed, async-ingesting RAG app.

## What's here

- `backend/devices.py`, `push.py` — device/push-token store and Expo push sending (reused as-is from the mobile/push pattern established earlier in the course).
- `backend/jobs.py` — ingestion (HOS 3) plus a push notification on completion.
- `backend/main.py` — adds `POST /register-device`, and CORS middleware (needed only for testing via `npx expo start --web`, which runs in a real browser — Streamlit and Expo Go on a device don't need it).
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
cp -n .env.example .env   # devcontainer does this automatically; harmless if it already ran
npx expo start --tunnel
```

Set `EXPO_PUBLIC_API_BASE` in `mobile/.env` to wherever your backend is reachable from your phone — in Codespaces, port 8000's forwarded URL (make it **Public** in the Ports tab first), e.g. `EXPO_PUBLIC_API_BASE=https://your-codespace-name-8000.app.github.dev`. Reload the app on your phone after editing (shake it → Reload) — saving alone doesn't apply a `.env` change.

**In Codespaces, always use `--tunnel`** — the plain command uses LAN mode, which can't work since your phone and the Codespace aren't on the same network (it looks like it's hanging, not actually connecting). The first time you ever run `--tunnel`, Expo installs a small tunneling package on the spot, which takes a minute or two — after that, it starts fast. Running locally with your phone on the same Wi-Fi as your laptop? Drop `--tunnel`, it's faster.

Push notifications need one extra one-time step: run `npx eas init` inside `mobile/` (free Expo account, no credit card — see the [Account & Service Setup Guide](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/account-setup-guide.html#expo)). `getExpoPushTokenAsync()` requires an EAS project ID as of SDK 49+; without it, `App.js` now logs a warning and returns instead of crashing — chat still works fine either way.

Separately, current Expo Go on iOS also requires the same Expo account logged in on both your terminal (`npx expo login`) and the Expo Go app on your phone (tap the avatar icon on its home screen → sign in) just to load the project at all — this blocks chat too, not just push. Not yet required on Android.

Once it's running: replies over the phone will feel slower than a `curl` test from inside the Codespace terminal — expected, not a bug. The phone's request travels over the real internet through GitHub's port-forwarding proxy to reach the Codespace and back, on top of whatever the agentic loop itself takes.

Verified this session: `/health`, RAG-grounded chat, tool calling, `/register-device`, and a full async-ingestion-with-push job — all live, with Langfuse keys unset (confirmed it disables gracefully rather than crashing). `score_faithfulness()` unit-tested against a grounded answer (0.75), a deliberately invented one (0.077), and a plain "I don't know" (1.0 — nothing to check). `mobile/App.js` was code-reviewed (not run live against a real device this session) — the missing-projectId fix is based on Expo's documented SDK 49+ requirement, not a live repro.

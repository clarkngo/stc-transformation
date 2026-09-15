# Week 8 Starter — Mobile Client & Real-Time Push Notifications

Guide: [`../../week-08-mobile-push.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/week-08-mobile-push.html)

## What's already working

- Async ingestion (Week 7, solved)
- `backend/devices.py` + `backend/push.py` — device token storage and Expo push sending, both complete
- `backend/main.py` — a working `/register-device` endpoint, and CORS middleware for testing via `npx expo start --web` (not needed for Streamlit or Expo Go on a real device)
- `mobile/App.js` — a working chat screen wired to your backend, plus notification-permission requesting already implemented

## What you'll build this week

1. `mobile/App.js` — follow the `# TODO(week8)` comment: call `registerDevice()` (already written in `mobile/src/api.js`) once you have a push token.
2. `backend/jobs.py` — follow the `# TODO(week8)` comment: send a push notification when `ingest_document_job` finishes.

## Run it

**Recommended: GitHub Codespaces.** Push this folder to its own repo, then **Code → Codespaces → Create codespace on main** — `.devcontainer/devcontainer.json` installs both `backend/requirements.txt` and `mobile/`'s npm packages automatically, and creates both `backend/.env` and `mobile/.env` from their `.env.example` templates. Add your keys to `backend/.env`, then:

```bash
# Terminal 1 — API
cd backend && uvicorn main:app --reload

# Terminal 2 — worker
cd backend && python worker.py

# Terminal 3 — mobile (note --tunnel, see below)
cd mobile && npx expo start --tunnel
```

Two things are different from running locally, because your phone and the codespace aren't on the same network:

1. **Make the API port public.** In VS Code's **Ports** tab, find port `8000`, right-click it, and set its visibility to **Public** (it defaults to private/authenticated, which a plain `fetch()` from the Expo app can't get through). Copy that port's forwarded URL and set it as `EXPO_PUBLIC_API_BASE` in `mobile/.env`, e.g. `EXPO_PUBLIC_API_BASE=https://your-codespace-name-8000.app.github.dev`. Reload the app on your phone afterward (shake it → Reload) — editing `.env` needs a reload to take effect, saving alone isn't enough.
2. **Use Expo's tunnel mode** (`--tunnel`, already in the command above) instead of the default LAN mode — it routes the connection through Expo's own relay servers, since your phone can't reach a Codespace directly over local Wi-Fi the way it could reach your laptop.

**Running locally instead?**
```bash
# Terminal 1 — API (see Week 7 for setup)
cd backend && ./venv/bin/uvicorn main:app --reload

# Terminal 2 — worker
cd backend && ./venv/bin/python worker.py

# Terminal 3 — mobile
cd mobile
npm install
npx expo start
```
`cp mobile/.env.example mobile/.env` if the devcontainer didn't already, then edit `mobile/.env` — set `EXPO_PUBLIC_API_BASE` to your laptop's LAN IP, not `localhost` (your phone can't resolve your laptop's `localhost`). Scan the QR code with the **Expo Go** app.

Either way: push notifications only work on a **real physical device**, not a simulator/emulator.

**One more one-time step (EAS project):** getting a push token requires an EAS project ID (Expo SDK 49+). Run `npx eas init` inside `mobile/` (free Expo account, no credit card — see the [Account & Service Setup Guide](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/account-setup-guide.html#expo)); it writes `extra.eas.projectId` into `app.json` for you. Without it, the app logs a warning and skips push registration instead of crashing — chat still works fine either way, and the guide's Common Snags section covers this too.

**And one more (Expo Go login):** current Expo Go on iOS requires the same Expo account logged in on both your terminal (`npx expo login`) and the Expo Go app on your phone (tap the avatar icon on its home screen → sign in) just to load the project at all — this blocks chat too, not just push. Not yet required on Android.

# Week 8 Starter — Mobile Client & Real-Time Push Notifications

Guide: [`../../week-08-mobile-push.html`](../../week-08-mobile-push.html)

## What's already working

- Async ingestion (Week 7, solved)
- `backend/devices.py` + `backend/push.py` — device token storage and Expo push sending, both complete
- `backend/main.py` — a working `/register-device` endpoint
- `mobile/App.js` — a working chat screen wired to your backend, plus notification-permission requesting already implemented

## What you'll build this week

1. `mobile/App.js` — follow the `# TODO(week8)` comment: call `registerDevice()` (already written in `mobile/src/api.js`) once you have a push token.
2. `backend/jobs.py` — follow the `# TODO(week8)` comment: send a push notification when `ingest_document_job` finishes.

## Run it

**Recommended: GitHub Codespaces.** Push this folder to its own repo, then **Code → Codespaces → Create codespace on main** — `.devcontainer/devcontainer.json` installs both `backend/requirements.txt` and `mobile/`'s npm packages automatically. Add your keys to `backend/.env`, then:

```bash
# Terminal 1 — API
cd backend && uvicorn main:app --reload

# Terminal 2 — worker
cd backend && python worker.py

# Terminal 3 — mobile (note --tunnel, see below)
cd mobile && npx expo start --tunnel
```

Two things are different from running locally, because your phone and the codespace aren't on the same network:

1. **Make the API port public.** In VS Code's **Ports** tab, find port `8000`, right-click it, and set its visibility to **Public** (it defaults to private/authenticated, which a plain `fetch()` from the Expo app can't get through). Copy that port's forwarded URL and set it as `API_BASE` in `mobile/src/api.js`, e.g. `https://your-codespace-name-8000.app.github.dev`.
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
Edit `mobile/src/api.js` — set `API_BASE` to your laptop's LAN IP, not `localhost` (your phone can't resolve your laptop's `localhost`). Scan the QR code with the **Expo Go** app.

Either way: push notifications only work on a **real physical device**, not a simulator/emulator.

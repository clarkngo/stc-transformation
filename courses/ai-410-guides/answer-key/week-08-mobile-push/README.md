# Week 8 Answer Key — Mobile Client & Real-Time Push Notifications

Guide: [`../../week-08-mobile-push.html`](../../week-08-mobile-push.html)
Starter: [`../../starter-code/week-08-mobile-push`](../../starter-code/week-08-mobile-push)

This is the **fully solved** version of Week 8 — everything a student
would build is already filled in, for SME/instructor verification
against a working reference. It's not what students should be given.

## What's solved here (vs. the starter)

1. `mobile/App.js` — calls `registerDevice()` once a push token is available (the starter leaves this commented out behind a `# TODO(week8)`).
2. `backend/jobs.py` — sends a push notification via `send_push_notification()` when `ingest_document_job` finishes (the starter leaves this commented out behind a `# TODO(week8)`).

Everything else (async ingestion from Week 7, device-token storage, the `/register-device` endpoint, the chat screen) was already complete in the starter and is unchanged here.

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

**One more one-time step:** `getExpoPushTokenAsync()` requires an EAS project ID as of Expo SDK 49+. Run `npx eas init` inside `mobile/` (free Expo account, no credit card — see the [Account & Service Setup Guide](../../account-setup-guide.html#expo)); it writes `extra.eas.projectId` into `app.json` for you. Without it, `App.js` now logs a warning and skips push registration instead of crashing — chat still works fine either way. (Fixed this session — the previous version called `getExpoPushTokenAsync()` with no arguments, which throws when no project ID is configured; not run live against a real device this session, the fix is based on Expo's documented SDK requirement.)

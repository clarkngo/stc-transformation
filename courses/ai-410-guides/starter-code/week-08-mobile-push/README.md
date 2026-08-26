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

Scan the QR code with the **Expo Go** app on your phone (no Xcode/Android Studio needed). Before testing, edit `mobile/src/api.js` — set `API_BASE` to your laptop's LAN IP, not `localhost` (your phone can't resolve your laptop's `localhost`).

Push notifications only work on a **real physical device**, not a simulator/emulator.

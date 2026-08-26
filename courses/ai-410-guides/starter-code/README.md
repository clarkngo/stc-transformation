# AI 410 — Starter Code

One folder per week, matching the [10 hands-on guides](../index.html). Each folder is a **complete, standalone snapshot** of the app as it exists at the *start* of that week — everything from prior weeks is already working, and that week's new piece is stubbed in (not solved) so the exercise in the guide still has something to do.

Because each week is a full snapshot rather than a diff, **you can copy any single week's folder straight into its own project** and it will run on its own — you don't need the other week folders alongside it.

```bash
cp -r week-04-vector-db-rag ~/projects/my-ai410-app
cd ~/projects/my-ai410-app
```

## Layout

| Folder | What it contains |
|---|---|
| `week-01-foundations/` | FastAPI backend with a stubbed `/chat` endpoint + a working vanilla HTML/JS chat frontend |
| `week-02-agentic-loop/` | + an agent loop skeleton with one dummy tool already wired |
| `week-03-embeddings-chunking/` | A standalone sandbox script + sample documents — not part of the main app |
| `week-04-vector-db-rag/` | + a vector DB schema and a `retrieve()` stub to fill in |
| `week-05-checkpoint-architecture-review/` | No code — a diagramming/review template only |
| `week-06-guardrails/` | + a validation/retry middleware template |
| `week-07-async-processing/` | + queue and worker boilerplate |
| `week-08-mobile-push/` | + an Expo (React Native) mobile shell wired to the backend |
| `week-09-observability/` | + Langfuse installed and configured, instrumentation points marked |
| `week-10-capstone-deploy/` | + a Dockerfile and deploy config |

## Before you start any week

1. Read that week's guide first (`../week-0N-*.html`) — the code here is the "Scaffold Provided" from that guide, not a solution.
2. Copy `.env.example` to `.env` and fill in your own API keys. Never commit `.env`.
3. Each folder's own `README.md` lists exactly what's already working and what you're expected to build.

## Common setup (Python weeks)

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
uvicorn main:app --reload
```

## Common setup (Week 8 mobile)

```bash
cd mobile
npm install
npx expo start
```
Scan the QR code with the **Expo Go** app on your phone — no Xcode or Android Studio needed.

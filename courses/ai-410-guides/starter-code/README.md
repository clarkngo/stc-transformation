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

## Recommended: run every week in GitHub Codespaces

Every week's folder (except Week 5, which has no code) includes a `.devcontainer/devcontainer.json`, so opening it as a Codespace gets you a ready-to-go environment automatically — dependencies installed, `.env` created from `.env.example`, ports forwarded — with nothing else on the machine to conflict with it.

```bash
# 1. Copy the week you want into its own repo (or push the whole
#    starter-code/ tree to one repo, and open individual week
#    subfolders as needed).
# 2. On GitHub: Code → Codespaces → Create codespace on main.
# 3. Wait for setup to finish, then add your real API keys to the
#    .env file it created for you.
```

This matters more than it might seem: the single most common bug students hit in this course is a local machine that already has an old or conflicting version of a package (`google-genai`, `uvicorn`, etc.) installed globally under a different Python, which silently gets used instead of the project's own dependencies — confusing errors that have nothing to do with your actual code. A Codespace is a brand-new container with nothing pre-installed to conflict with, so this entire class of problem doesn't happen. It's also free for the amount of usage this course needs, and works identically whether you're on a high-end laptop or a Chromebook.

Once you're in a Codespace, run the same commands the week's own README shows — just without any `./venv/bin/` prefix or `source venv/bin/activate` step, since there's no need for a virtual environment inside an already-isolated container:

```bash
cd backend
uvicorn main:app --reload
```

## Running locally instead

If you'd rather work on your own machine, every week's README has a "Running locally" section using a Python virtual environment, to keep this project's dependencies isolated from anything else on your system:

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
./venv/bin/uvicorn main:app --reload
```

Always call `./venv/bin/uvicorn` (and `./venv/bin/python` for scripts) rather than the bare command — if your system already has something like `uvicorn` or `google-genai` installed globally, an unactivated shell will silently run those instead of the ones in this project's venv.

## Week 8 mobile (either environment)

```bash
cd mobile
npm install
npx expo start          # add --tunnel if you're running from a Codespace
```
Scan the QR code with the **Expo Go** app on your phone — no Xcode or Android Studio needed. See Week 8's own README for the extra step needed when the backend is running in a Codespace (making the port public) rather than on your own machine.

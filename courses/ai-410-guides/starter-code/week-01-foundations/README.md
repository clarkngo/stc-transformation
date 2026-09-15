# Week 1 Starter — Foundations of Full-Stack AI Systems

Guide: [`../../week-01-foundations.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/week-01-foundations.html)

## What's already working

- A FastAPI backend (`backend/main.py`) with a `/chat` endpoint
- A Streamlit chat page (`frontend/app.py`) already wired to call `/chat` and display the reply

## What you'll build this week

The `/chat` endpoint currently returns a **hardcoded stub reply**. Your job is to replace it with a real call to the Gemini API. Look for the `# TODO(week1)` comment in `backend/main.py`.

Get a free API key (no credit card required) at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Run it

### Option A: GitHub Codespaces (recommended)

1. Push this folder to its own GitHub repo (or a repo that contains it).
2. On the repo page: **Code → Codespaces → Create codespace on main**.
3. Wait for setup to finish — `postCreateCommand` in `.devcontainer/devcontainer.json` automatically installs `backend/requirements.txt` and `frontend/requirements.txt`, and creates `backend/.env` for you.
4. Open `backend/.env` and paste in your `GEMINI_API_KEY`.
5. In one terminal, run the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   No venv, no `./venv/bin/...` — the codespace container is already an isolated environment with nothing else installed in it, so there's nothing for `uvicorn` to conflict with.
6. Open a **second terminal** and run the frontend:
   ```bash
   cd frontend
   streamlit run app.py
   ```
7. Codespaces auto-forwards port 8501 and opens a preview tab for it (that's what `"onAutoForward": "openPreview"` in `.devcontainer/devcontainer.json` does). If it doesn't pop up, open it from the **Ports** tab.

**Why Codespaces:** every student gets an identical, disposable environment — no local Python version conflicts, no pre-existing global packages to accidentally shadow this project's dependencies, and it works the same on a Chromebook as it does on a high-end laptop. It's free for the amount of usage this course needs. It's also why the frontend is a Streamlit app and not a static `index.html`: a plain `file://` page doesn't play well with Codespaces' port forwarding, while Streamlit runs as its own server with a real forwarded URL.

### Option B: Run locally

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/uvicorn main:app --reload
```

Using `./venv/bin/uvicorn` (not just `uvicorn`) matters here — if your system already has a `uvicorn`/`google-genai` installed globally, an unactivated shell will silently run those instead of the ones in this project's venv. This entire class of problem is exactly what Option A avoids.

In a second terminal, run the frontend the same way:

```bash
cd frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

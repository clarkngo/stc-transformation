# Week 1 Starter — Foundations of Full-Stack AI Systems

Guide: [`../../week-01-foundations.html`](../../week-01-foundations.html)

## What's already working

- A FastAPI backend (`backend/main.py`) with a `/chat` endpoint
- A vanilla HTML/JS chat page (`frontend/index.html` + `frontend/script.js`) already wired to call `/chat` and display the reply
- CORS is already configured — don't remove it

## What you'll build this week

The `/chat` endpoint currently returns a **hardcoded stub reply**. Your job is to replace it with a real call to the Claude API. Look for the `# TODO(week1)` comment in `backend/main.py`.

## Run it

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

Then open `frontend/index.html` directly in your browser (double-click it, or `open frontend/index.html` on macOS).

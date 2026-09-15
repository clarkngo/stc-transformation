# Week 1 Answer Key — Foundations of Full-Stack AI Systems

Guide: [`../../week-01-foundations.html`](../../week-01-foundations.html)
Starter: [`../../starter-code/week-01-foundations`](../../starter-code/week-01-foundations)

Reference solution for SME/instructor verification — not what students should be given.

## What's solved here (vs. the starter)

`backend/main.py` — the `/chat` endpoint calls Gemini for real (`client.interactions.create(...)`) instead of returning the hardcoded stub reply. The starter leaves this as a stub with a `# TODO(week1)` comment.

## Run it

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/uvicorn main:app --reload
```

In a second terminal:

```bash
cd frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

Ask it anything and confirm you get a real, on-topic Gemini-generated answer, not a canned reply.

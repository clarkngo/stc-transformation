# HOS 1 Answer Key — Foundations & the Agentic Loop

Guide: [`../../hos-01-foundations-agentic-loop.html`](https://clarkngo.github.io/stc-transformation/courses/ai-410-guides/hos-01-foundations-agentic-loop.html)
Starter: [`../../starter-code/hos-01-foundations-agentic-loop`](../../starter-code/hos-01-foundations-agentic-loop)

This is a **reference solution**, for SME/instructor verification against a working example — not what students should be given. Unlike the week-by-week starter code, this HOS's starter is a near-blank seed (students scaffold the app themselves with an AI assistant), so there's no single "correct" Create/Scaffold output to diff against. This answer key represents one reasonable result of that stage, plus the Understand & Refine addition and reference notes for the two written stages.

## What's here

- `backend/` — FastAPI backend with `/chat` (agentic loop) and `/health`. `tools.py` has **two** tools: `calculate` (what a typical Create/Scaffold pass produces) and `word_count` (written by hand, representing Understand & Refine).
- `frontend/` — Streamlit chat client, same pattern as the rest of the course.
- `EVALUATE.md` — reference notes on what a genuine Evaluate pass should surface (not a checklist to hand to students).
- `ANALYZE.md` — a reference walkthrough of the function-call detection logic in `agent.py`, for judging the depth of a student's own explanation.

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

Try a message that needs the calculator ("what's 342 times 87?"), one that needs word counting ("how many words are in 'the quick brown fox jumps'?"), and one that needs neither — confirm the agent only calls a tool when it actually helps.

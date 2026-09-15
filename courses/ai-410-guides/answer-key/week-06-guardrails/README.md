# Week 6 Answer Key — Probabilistic Logic & Guardrails

Guide: [`../../week-06-guardrails.html`](../../week-06-guardrails.html)
Starter: [`../../starter-code/week-06-guardrails`](../../starter-code/week-06-guardrails)

Reference solution for SME/instructor verification — not what students should be given. Built on top of Week 4's RAG app.

## What's solved here (vs. the starter)

- `backend/guardrails.py`'s `ToolArgs` schema — guards the `calculate` tool's arguments (`expression: str`), catching a missing or malformed value before it reaches the tool function.
- `backend/agent.py` — wraps the `calculate` tool call with `call_with_guardrail()`. The `ping` tool is deliberately **not** wrapped — it takes no arguments, so guarding it with `ToolArgs` would make every `ping` call fail validation.

## Run it

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/python ingest.py
./venv/bin/uvicorn main:app --reload
```

In a second terminal:

```bash
cd frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

Feed the agent a prompt likely to produce malformed tool arguments and confirm the guardrail catches it. Also confirm "call your ping tool" still works — a common mistake is guarding every tool call unconditionally, which breaks `ping`.

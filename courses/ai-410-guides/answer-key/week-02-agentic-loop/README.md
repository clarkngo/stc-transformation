# Week 2 Answer Key — Structured Outputs & the Agentic Loop

Guide: [`../../week-02-agentic-loop.html`](../../week-02-agentic-loop.html)
Starter: [`../../starter-code/week-02-agentic-loop`](../../starter-code/week-02-agentic-loop)

Reference solution for SME/instructor verification — not what students should be given.

## What's solved here (vs. the starter)

`backend/tools.py` — registers a real `calculate` tool (safe AST-based arithmetic, not a raw `eval()`) alongside the starter's dummy `ping` tool, and it's wired into `TOOLS`/`TOOL_FUNCTIONS`. The starter leaves the `calculate` tool commented out behind `# TODO(week2)` comments.

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

Ask something that needs the calculator ("what's 342 times 87?") and something that doesn't — confirm the agent only calls the tool when it actually needs to.

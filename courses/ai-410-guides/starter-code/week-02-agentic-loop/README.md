# Week 2 Starter — Structured Outputs & the Agentic Loop

Guide: [`../../week-02-agentic-loop.html`](../../week-02-agentic-loop.html)

## What's already working

- Week 1's real Claude call (`backend/main.py`)
- A complete agentic loop in `backend/agent.py`: send message + tools → check for `tool_use` → run the matching Python function → send the result back → get a final answer
- One dummy tool already registered in `backend/tools.py` (`ping`) so you can see the whole loop fire before writing your own tool

## What you'll build this week

Add a **real** tool. Open `backend/tools.py` and follow the `# TODO(week2)` comment: define a new tool schema (e.g. a calculator) and the Python function that executes it, then register it in the `TOOLS` list.

## Run it

Same as Week 1:
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

Try asking something that needs your new tool, and something that doesn't — confirm the agent only calls the tool when it actually needs to.

# Week 2 Starter — Structured Outputs & the Agentic Loop

Guide: [`../../week-02-agentic-loop.html`](../../week-02-agentic-loop.html)

## What's already working

- Week 1's real Gemini call (`backend/main.py`)
- A complete agentic loop in `backend/agent.py`: send message + tools → check for a `function_call` step → run the matching Python function → send the result back → get a final answer
- One dummy tool already registered in `backend/tools.py` (`ping`) so you can see the whole loop fire before writing your own tool

## What you'll build this week

Add a **real** tool. Open `backend/tools.py` and follow the `# TODO(week2)` comment: define a new tool schema (e.g. a calculator) and the Python function that executes it, then register it in the `TOOLS` list.

## Run it

**Recommended: GitHub Codespaces.** Push this folder to its own repo, then **Code → Codespaces → Create codespace on main** — setup (installing `backend/requirements.txt`, creating `backend/.env`) happens automatically via `.devcontainer/devcontainer.json`. Add your `GEMINI_API_KEY` to `backend/.env`, then:
```bash
cd backend
uvicorn main:app --reload
```

**Running locally instead?**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
./venv/bin/uvicorn main:app --reload
```

Try asking something that needs your new tool, and something that doesn't — confirm the agent only calls the tool when it actually needs to.

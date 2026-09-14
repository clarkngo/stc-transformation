"""
AI 410 — Week 2 starter: Structured Outputs & the Agentic Loop

Week 1's stub is now a real Gemini call, wired through the agentic
loop in agent.py. Your job this week is in tools.py — add a real tool
the agent can decide to use.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_agent

load_dotenv()

app = FastAPI(title="AI 410 — Week 2")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    reply = run_agent(body.message)
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}

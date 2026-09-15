"""
AI 410 — Week 2 answer key: Structured Outputs & the Agentic Loop

Week 1's stub is now a real Gemini call, wired through the agentic
loop in agent.py, with a real "calculate" tool registered alongside
the dummy "ping" tool.
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
    return ChatResponse(reply=run_agent(body.message))


@app.get("/health")
def health():
    return {"status": "ok"}

"""
AI 410 — Week 6 answer key: Probabilistic Logic & Guardrails

RAG is fully wired (Week 4). The `calculate` tool's arguments are now
guarded with a Pydantic schema + retry, per guardrails.py.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_agent

load_dotenv()

app = FastAPI(title="AI 410 — Week 6")


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

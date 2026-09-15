"""
HOS 1 answer key: Foundations & the Agentic Loop.

FastAPI backend exposing the agent over HTTP. The frontend (Streamlit)
calls this server-side, so no CORS middleware is needed.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_agent

load_dotenv()

app = FastAPI(title="AI 410 — HOS 1")


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

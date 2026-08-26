"""
AI 410 — Week 1 starter: Foundations of Full-Stack AI Systems

This is the stub. The /chat endpoint returns a hardcoded reply so you
can confirm the frontend <-> backend wiring works before touching the
LLM call itself. Your job this week: replace the stub with a real
call to Claude. See the TODO below.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI 410 — Week 1")

# Don't remove this — the frontend runs from a plain file:// page,
# which needs CORS enabled to call this API at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    # TODO(week1): replace this stub with a real call to Claude.
    #
    # from anthropic import Anthropic
    # client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    #
    # response = client.messages.create(
    #     model="claude-sonnet-5",
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": body.message}],
    # )
    # return ChatResponse(reply=response.content[0].text)

    return ChatResponse(reply=f"(stub reply) You said: {body.message!r}")


@app.get("/health")
def health():
    return {"status": "ok"}

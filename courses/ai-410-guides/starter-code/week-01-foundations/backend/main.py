"""
AI 410 — Week 1 starter: Foundations of Full-Stack AI Systems

This is the stub. The /chat endpoint returns a hardcoded reply so you
can confirm the frontend <-> backend wiring works before touching the
LLM call itself. Your job this week: replace the stub with a real
call to Gemini. See the TODO below.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI 410 — Week 1")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    # TODO(week1): replace this stub with a real call to Gemini.
    #
    from google import genai
    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    
    interaction = client.interactions.create(
        model="gemini-flash-latest",
        input=body.message,
    )
    return ChatResponse(reply=interaction.output_text)

    # return ChatResponse(reply=f"(stub reply) You said: {body.message!r}")


@app.get("/health")
def health():
    return {"status": "ok"}

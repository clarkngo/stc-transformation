"""
AI 410 — Week 1 answer key: Foundations of Full-Stack AI Systems

The /chat endpoint makes a real call to Gemini.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI 410 — Week 1")
client = genai.Client()  # reads GEMINI_API_KEY from the environment


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    interaction = client.interactions.create(
        model="gemini-flash-latest",
        input=body.message,
    )
    return ChatResponse(reply=interaction.output_text)


@app.get("/health")
def health():
    return {"status": "ok"}

"""
AI 410 — Week 10 starter: Capstone Integration & Deployment.

Everything from Weeks 1-9 (chat, RAG, guardrails, async ingestion,
push notifications, tracing) is already solved and carried forward
unchanged. This week is integration and deployment — see the guide
and README for what to do.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rq.job import Job

from agent import run_agent
from devices import register_push_token
from jobs import ingest_document_job
from queue_setup import queue, redis_conn

load_dotenv()

app = FastAPI(title="AI 410 — Week 10")

# CORS — needed for testing with `npx expo start --web` (Expo Go's
# browser-based mode). Streamlit and Expo Go on a real device never hit
# this: they call the backend server-side / outside a browser, so CORS
# doesn't apply to them. Expo web does run in a real browser though, and
# a browser blocks a cross-origin fetch() until the server explicitly
# allows it (a preflight OPTIONS check) — that's what this middleware
# answers. Wide open (allow_origins=["*"]) is fine for a course dev
# environment; a real product would scope this to its actual frontend's
# domain instead.
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


class UploadRequest(BaseModel):
    doc_path: str
    device_id: str | None = None


class RegisterDeviceRequest(BaseModel):
    device_id: str
    push_token: str


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    return ChatResponse(reply=run_agent(body.message))


@app.post("/documents")
def upload_document(body: UploadRequest):
    job = queue.enqueue(ingest_document_job, body.doc_path, body.device_id)
    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    return {"status": job.get_status(), "result": job.result}


@app.post("/register-device")
def register_device(body: RegisterDeviceRequest):
    register_push_token(body.device_id, body.push_token)
    return {"status": "registered"}


@app.get("/health")
def health():
    return {"status": "ok"}

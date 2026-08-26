"""
AI 410 — Week 8 starter: Mobile Client & Push Notifications

Async ingestion (Week 7) is solved below. This week adds device
registration for push notifications — already working. Your job is
in jobs.py: trigger a push when a job finishes (see the TODO there),
and in the mobile/ app: request permission and register the device.
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

app = FastAPI(title="AI 410 — Week 8")

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
